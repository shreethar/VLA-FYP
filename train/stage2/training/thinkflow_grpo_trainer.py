import os
import copy
import torch
from trl import GRPOTrainer
from trl.extras.profiling import profiling_context
from trl.models import unwrap_model_for_generation
from trl.data_utils import prepare_multimodal_messages

class ThinkFlowGRPOTrainer(GRPOTrainer):
    """
    A custom GRPOTrainer that seamlessly integrates with ThinkFlow-VLA's decoupled distillation.
    It hooks into `_generate_single_turn` to optimize generation and bypass prompt duplication,
    and hooks into `compute_loss` to extract the </think> hidden state (h_T) on the fly,
    saving the rollout buffer to disk exactly as `train_stage2.py`'s `teacher_only` mode does.
    """
    def __init__(self, *args, thinkflow_offline_dir=None, think_end_token_id=None, tf_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.thinkflow_offline_dir = thinkflow_offline_dir
        self.think_end_token_id = think_end_token_id
        self.tf_tokenizer = tf_tokenizer
        self.micro_step_counter = 0

        if self.thinkflow_offline_dir:
            os.makedirs(self.thinkflow_offline_dir, exist_ok=True)

    def _generate_single_turn(self, prompts, images):
        device = self.accelerator.device
        
        # Conversational data prep
        kwargs = {}
        if images is not None:
            kwargs = {"images": images}
            for prompt, image_list in zip(prompts, images):
                if isinstance(prompt, list):
                    prepare_multimodal_messages(prompt, num_images=len(image_list))
                    
        prompts_text = []
        for prompt in prompts:
            if isinstance(prompt, list):
                prompts_text.append(
                    self.processing_class.apply_chat_template(
                        prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                )
            else:
                prompts_text.append(prompt)
                
        # Deduplicate prompts and images (since they are repeated num_generations times)
        G = self.num_generations
        unique_prompts_text = prompts_text[::G]
        unique_images = images[::G] if images is not None else None
        
        unique_kwargs = {}
        if unique_images is not None:
            unique_kwargs["images"] = unique_images
            
        # Process unique inputs using the processor
        generate_inputs = self.processing_class(
            text=unique_prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **unique_kwargs,
        )
        
        # Prepare inputs (move to device)
        generate_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in generate_inputs.items()
        }
        
        # Configure generation to return G sequences per prompt
        generation_config = copy.deepcopy(self.generation_config)
        generation_config.num_return_sequences = G
        generation_config.do_sample = True  # crucial to sample and get G distinct pathways
        
        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
        ):
            # Generate the G completions per prompt in a single call!
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, generation_config=generation_config, disable_compile=True
            )
            
        # Extract prompt ids and prompt masks
        prompt_ids = generate_inputs["input_ids"]
        prompt_mask = generate_inputs["attention_mask"]
        
        # Repeat the prompt ids and prompt masks G times to match the B * G shape
        prompt_ids = prompt_ids.repeat_interleave(G, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(G, dim=0)
        
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        prompt_ids_list = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
        completion_ids_list = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
        logprobs = None
        
        # Reconstruct forward_kwargs for all B * G outputs
        forward_kwargs = {}
        
        # 1. Handle pixel_values (flattened patches) and image_grid_thw
        if "pixel_values" in generate_inputs:
            pv = generate_inputs["pixel_values"]
            thw = generate_inputs["image_grid_thw"]
            
            # Repeat thw
            repeated_thw = thw.repeat_interleave(G, dim=0)
            
            # Repeat pv based on dynamic patch count per image
            num_patches = thw.prod(dim=-1)
            pv_list = []
            start = 0
            for n in num_patches:
                n = n.item()
                img_patches = pv[start : start + n]
                for _ in range(G):
                    pv_list.append(img_patches)
                start += n
            repeated_pv = torch.cat(pv_list, dim=0)
            
            forward_kwargs["pixel_values"] = repeated_pv
            forward_kwargs["image_grid_thw"] = repeated_thw
            
        # 2. Handle video features
        if "pixel_values_videos" in generate_inputs:
            pv_v = generate_inputs["pixel_values_videos"]
            thw_v = generate_inputs["video_grid_thw"]
            
            repeated_thw_v = thw_v.repeat_interleave(G, dim=0)
            num_patches_v = thw_v.prod(dim=-1)
            pv_v_list = []
            start = 0
            for n in num_patches_v:
                n = n.item()
                vid_patches = pv_v[start : start + n]
                for _ in range(G):
                    pv_v_list.append(vid_patches)
                start += n
            repeated_pv_v = torch.cat(pv_v_list, dim=0)
            
            forward_kwargs["pixel_values_videos"] = repeated_pv_v
            forward_kwargs["video_grid_thw"] = repeated_thw_v

        # 3. Repeat any other standard 1D/2D tensors
        for k, v in generate_inputs.items():
            if k not in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
                if isinstance(v, torch.Tensor):
                    forward_kwargs[k] = v.repeat_interleave(G, dim=0)
            
        return prompt_ids_list, completion_ids_list, logprobs, forward_kwargs

    def _generate_and_score_completions(self, inputs):
        # inputs is a List of dictionaries, size: (batch_size * steps_per_generation)
        output = super()._generate_and_score_completions(inputs)
        
        # We must repeat our custom dataset fields G times so they match the B*G shape
        # of the other tensors in output. This ensures `split_tensor_dict` handles them properly.
        fields_to_keep = [
            "ground_truth", "sample_id", 
            "pixel_values", "image_grid_thw",
            "pixel_values_videos", "video_grid_thw"
        ]
        
        for field in fields_to_keep:
            if field in inputs[0]:
                val_list = [example[field] for example in inputs for _ in range(self.num_generations)]
                output[f"original_{field}"] = val_list
            
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        # Right after computing loss (which happens per micro-batch), save the offline data
        if self.thinkflow_offline_dir is not None:
            self._save_thinkflow_offline_data(model, inputs)
            self.micro_step_counter += 1
            
        return loss

    def _save_thinkflow_offline_data(self, model, inputs):
        device = model.device

        B_times_G = inputs["prompt_ids"].shape[0]
        G = self.num_generations
        B = B_times_G // G

        advantages = inputs["advantages"] # [B*G]
        adv_reshaped = advantages.view(B, G)
        best_idx = adv_reshaped.argmax(dim=1)
        worst_idx = adv_reshaped.argmin(dim=1)
        
        batch_offsets = torch.arange(B, device=device) * G
        best_global_idx = batch_offsets + best_idx
        worst_global_idx = batch_offsets + worst_idx

        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_mask = inputs["completion_mask"]

        # 1. tau+ (best responses)
        tau_pos_prompt_ids = prompt_ids[best_global_idx]
        tau_pos_completion_ids = completion_ids[best_global_idx]
        tau_pos_ids = torch.cat([tau_pos_prompt_ids, tau_pos_completion_ids], dim=1)
        
        tau_pos_prompt_mask = prompt_mask[best_global_idx]
        tau_pos_completion_mask = completion_mask[best_global_idx]
        tau_pos_mask = torch.cat([tau_pos_prompt_mask, tau_pos_completion_mask], dim=1)

        # Response mask (0 for prompt, 1 for completion)
        pos_response_mask = torch.zeros_like(tau_pos_mask)
        pos_response_mask[:, tau_pos_prompt_ids.shape[1]:] = tau_pos_completion_mask

        # 2. tau- (worst responses)
        tau_neg_prompt_ids = prompt_ids[worst_global_idx]
        tau_neg_completion_ids = completion_ids[worst_global_idx]
        tau_neg_ids = torch.cat([tau_neg_prompt_ids, tau_neg_completion_ids], dim=1)
        
        tau_neg_prompt_mask = prompt_mask[worst_global_idx]
        tau_neg_completion_mask = completion_mask[worst_global_idx]
        tau_neg_mask = torch.cat([tau_neg_prompt_mask, tau_neg_completion_mask], dim=1)

        # 3. Dynamic hook for h_T (best reasoning state)
        think_end_positions = (tau_pos_ids == self.think_end_token_id).nonzero(as_tuple=True)[1]
        
        if len(think_end_positions) < B:
            # fallback if model fails to generate </think>
            think_end_positions = torch.full((B,), tau_pos_ids.shape[1] - 1, device=device)
        else:
            # handle cases where multiple </think> generated by taking the first one
            think_end_positions = think_end_positions.view(B, -1)[:, 0]

        forward_kwargs = {}
        if "original_pixel_values" in inputs and inputs["original_pixel_values"][0] is not None:
            pvs = [inputs["original_pixel_values"][i] for i in best_global_idx.tolist()]
            forward_kwargs["pixel_values"] = torch.stack(pvs).to(device) if pvs else None
        
        if "original_image_grid_thw" in inputs and inputs["original_image_grid_thw"][0] is not None:
            ig = [inputs["original_image_grid_thw"][i] for i in best_global_idx.tolist()]
            forward_kwargs["image_grid_thw"] = torch.stack(ig).to(device) if ig else None

        if "original_pixel_values_videos" in inputs and inputs["original_pixel_values_videos"][0] is not None:
            pvs_v = [inputs["original_pixel_values_videos"][i] for i in best_global_idx.tolist()]
            forward_kwargs["pixel_values_videos"] = torch.stack(pvs_v).to(device) if pvs_v else None
            
        if "original_video_grid_thw" in inputs and inputs["original_video_grid_thw"][0] is not None:
            ig_v = [inputs["original_video_grid_thw"][i] for i in best_global_idx.tolist()]
            forward_kwargs["video_grid_thw"] = torch.stack(ig_v).to(device) if ig_v else None

        with torch.no_grad():
            unwrapped_model = self.accelerator.unwrap_model(model)
            outputs = unwrapped_model(
                input_ids=tau_pos_ids,
                attention_mask=tau_pos_mask,
                output_hidden_states=True,
                use_cache=False,
                **forward_kwargs
            )
            h_all = outputs.hidden_states[-1] # [B, seq, d_model]
            h_T = h_all[torch.arange(B, device=device), think_end_positions] # [B, d_model]

        ground_truth = None
        if "original_ground_truth" in inputs:
            gt_dicts = [inputs["original_ground_truth"][i] for i in best_global_idx.tolist()]
            if gt_dicts and gt_dicts[0] is not None:
                ground_truth = {k: [] for k in gt_dicts[0].keys()}
                for gt in gt_dicts:
                    for k, v in gt.items():
                        ground_truth[k].append(v)
                for k, v in ground_truth.items():
                    if isinstance(v[0], torch.Tensor):
                        ground_truth[k] = torch.stack(v, dim=0)
                    elif k == "gt_waypoints":
                        ground_truth[k] = torch.tensor(v, dtype=torch.float32)

        sample_ids = None
        if "original_sample_id" in inputs:
            sample_ids = [inputs["original_sample_id"][i] for i in best_global_idx.tolist()]

        step = self.state.global_step + 1
        accum_idx = self.micro_step_counter % self.args.gradient_accumulation_steps
        rewards = torch.zeros(G, B, dtype=torch.float32)

        # Match exact dictionary keys from train_stage2.py's teacher_only mode
        data_to_save = {
            "global_step":           step,
            "micro_step":            accum_idx,
            "sample_ids":            sample_ids,

            "input_ids":             prompt_ids[batch_offsets].cpu(),
            "attention_mask":        prompt_mask[batch_offsets].cpu(),
            
            "image_grid_thw":        forward_kwargs.get("image_grid_thw", torch.tensor([])).cpu() if "image_grid_thw" in forward_kwargs else None,
            "video_grid_thw":        forward_kwargs.get("video_grid_thw", torch.tensor([])).cpu() if "video_grid_thw" in forward_kwargs else None,
            "pixel_values":          forward_kwargs.get("pixel_values", torch.tensor([])).cpu() if "pixel_values" in forward_kwargs else None,
            "pixel_values_videos":   forward_kwargs.get("pixel_values_videos", torch.tensor([])).cpu() if "pixel_values_videos" in forward_kwargs else None,

            "gt_waypoints":          ground_truth["gt_waypoints"].cpu() if ground_truth and "gt_waypoints" in ground_truth else None,
            "ground_truth":          ground_truth,

            "tau_pos_ids":           tau_pos_ids.cpu(),
            "tau_pos_mask":          tau_pos_mask.cpu(),
            "tau_neg_ids":           tau_neg_ids.cpu(),
            "tau_neg_mask":          tau_neg_mask.cpu(),
            "tau_pos_response_mask": pos_response_mask.cpu(),
            
            "h_T":                   h_T.cpu(),

            "rewards":               rewards.cpu(),
            "advantages":            advantages.cpu(),
        }

        # Save decoupled tensor buffer
        filename = f"step_{step:06d}_micro_{accum_idx:02d}.pt"
        filepath = os.path.join(self.thinkflow_offline_dir, filename)
        torch.save(data_to_save, filepath)
