#!/usr/bin/env python3
"""
RoboVQA Evaluation Script

This script evaluates a Vision-Language-Action (VLA) model on the RoboVQA validation dataset.
It loads a HuggingFace checkpoint, processes the validation TFRecords, generates answers for
the VQA tasks, and computes the BLEU score using `sacrebleu`.
"""

import argparse
import os
import re
import sys
import random
from typing import List, Tuple, Dict, Any

# Disable GPU for TensorFlow immediately to prevent it from pre-allocating GPU memory
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import torch
import sacrebleu
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from absl import logging

# Disable tensorflow logs to keep output clean unless needed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.set_verbosity(logging.ERROR)

STAGE2_QA_SYSTEM = (
    "You are a robot manipulation assistant. Answer questions about robot tasks, "
    "object affordances, spatial relationships, and manipulation strategies based "
    "on the provided image or video frame. "
    "If reasoning, think step-by-step. "
    "Finally, output the answer after </think> without wrapping it in any brackets or tags."
)

# ==============================================================================
# Task Parsing Utilities (ported from RoboVQA Evaluation notebook)
# ==============================================================================

class Task:
    """A class for handling tags and splits in a given task."""

    PRED_STARTS = ['Robot:', 'Thought:', 'Action:']
    NOPRED_STARTS = ['User:', 'System:']

    PRED_START = '<PRED>'
    PRED_END = '</PRED>'
    PRED_ANSWER_BINARY_START = '<PRED:ANSWER:BINARY>'
    PRED_ANSWER_BINARY_END = '</PRED:ANSWER:BINARY>'
    PRED_ANSWER_DISCRETE_START = '<PRED:ANSWER:DISCRETE>'
    PRED_ANSWER_DISCRETE_END = '</PRED:ANSWER:DISCRETE>'
    PRED_ANSWER_START = '<PRED:ANSWER'
    PRED_ANSWER_END = '</PRED:ANSWER'
    PRED_ALL_START = '<PRED:'
    PRED_ALL_END = '</PRED:'

    TAGS_RE = r'(</*\w[:\w]*>)'

    def __init__(self, text: str):
        self.text = text

    def get_splits(self, split_type: str = 'speaker') -> List[Tuple[str, str]]:
        """Returns a list of (source, target) split pairs."""
        if split_type == 'pred':
            return self.get_splits_from_tags(
                start_tags=[self.PRED_START], end_tags=[self.PRED_END])
        elif split_type == 'binary':
            return self.get_splits_from_tags(
                start_tags=[self.PRED_ANSWER_BINARY_START], end_tags=[self.PRED_ANSWER_BINARY_END])
        elif split_type == 'discrete':
            return self.get_splits_from_tags(
                start_tags=[self.PRED_ANSWER_DISCRETE_START],
                end_tags=[self.PRED_ANSWER_DISCRETE_END])
        elif split_type == 'answer':
            return self.get_splits_from_tags(
                start_tags=[self.PRED_ANSWER_START], end_tags=[self.PRED_ANSWER_END])
        elif split_type == 'A:':
            return self.get_splits_from_tags(start_tags=['A:'], end_tags=[])
        elif split_type == 'speaker':
            return self.get_splits_from_tags(
                start_tags=self.PRED_STARTS, end_tags=self.NOPRED_STARTS)
        elif split_type == 'all':
            return self.get_splits_from_tags(
                start_tags=[self.PRED_ALL_START], end_tags=[self.PRED_ALL_END])
        else:
            raise ValueError(f'Unknown split type: {split_type}')

    def get_splits_from_tags(self, start_tags: List[str], end_tags: List[str]) -> List[Tuple[str, str]]:
        """Returns a list of (source, target) split pairs given start/end tags."""
        split_positions = []
        position = 0
        while position < len(self.text):
            start_position = self.find_next_tag(position, start_tags)
            if start_position is None:
                break
            end_position = self.find_next_tag(start_position, end_tags)
            if end_position is None:
                end_position = len(self.text)
            split_positions.append((start_position, end_position))
            position = end_position + 1
        return self.get_splits_from_positions(split_positions)

    def get_splits_from_positions(self, split_positions: List[Tuple[int, int]]) -> List[Tuple[str, str]]:
        """Returns a list of (source, target) split pairs given split positions."""
        splits = []
        for (split_position, end_position) in split_positions:
            source = ''
            if split_position > 0:
                source = self.text[:split_position]
                source = self._remove_tags(source)
            target = self.text[split_position:end_position]
            target = self._remove_tags(target)
            splits.append((source, target))

        if not splits:
            splits = [('', self.text)]

        return splits

    def find_next_tag(self, position: int, tags: List[str]) -> Any:
        tag_position = None
        lower_text = self.text.lower()
        for tag in tags:
            p = lower_text.find(tag.lower(), position)
            if p >= 0 and (tag_position is None or p < tag_position):
                tag_position = p
        return tag_position

    def _remove_tags(self, text: str) -> str:
        return re.sub(self.TAGS_RE, '', text)


class Tasks:
    """A class for handling and holding tasks information."""

    TASK_RE = r'(<task[:\w]*>)'
    RE_FLAGS = re.IGNORECASE

    def __init__(self, tasks_raw: str = None):
        self.tasks_dict = {}
        self.tasks_list = []
        self.tasks_types = []
        self.tasks_raw = tasks_raw
        if tasks_raw is not None:
            self.add(tasks_raw)

    def add(self, tasks: str):
        task_dict = self.text_to_dict(tasks)
        for name, ts in task_dict.items():
            if name not in self.tasks_dict:
                self.tasks_dict[name] = []
            self.tasks_dict[name].extend(ts)
            self.tasks_list.extend(ts)
            self.tasks_types.extend([name] * len(ts))

    def text_to_dict(self, text: str) -> Dict[str, List[str]]:
        """Splits raw serialized task string into dict mapping task tags to task bodies."""
        split = re.split(self.TASK_RE, text, flags=self.RE_FLAGS)[1:]
        tasks_dict = {}
        i = 0
        while i < len(split) - 1:
            tag = split[i].strip()
            task = split[i+1].lstrip()
            if task:
                if tag not in tasks_dict:
                    tasks_dict[tag] = []
                tasks_dict[tag].append(task)
            i += 2
        return tasks_dict


def fetch_question_answer(text: str) -> List[Tuple[int, str, str, str]]:
    """Extracts task index, task type, question, and answer from raw text."""
    tasks = Tasks(text)
    results = []
    for i, (task_type, ts) in enumerate(tasks.tasks_dict.items()):
        for task in ts:
            t = Task(task)
            splits = t.get_splits('A:')
            for split in splits:
                question, answer = split
                results.append((i, task_type, question.strip(), answer.strip()))
    return results


# ==============================================================================
# Model Evaluation Logic
# ==============================================================================

def clean_prediction(pred_raw: str) -> str:
    """Cleans up the raw model prediction to isolate the final answer string."""
    final_answer = pred_raw

    # Remove reasoning traces if the model output format has a </think> separator
    if '</think>\n\n' in final_answer:
        final_answer = final_answer.split('</think>\n\n', 1)[1]
    elif '</think>' in final_answer:
        final_answer = final_answer.split('</think>', 1)[1]

    final_answer = final_answer.strip()

    # Strip custom end-of-sequence tags
    if final_answer.endswith('<|im_end|>'):
        final_answer = final_answer[:-len('<|im_end|>')].strip()

    # Strip potential prefix strings
    if final_answer.lower().startswith('assistant:'):
        final_answer = final_answer[len('assistant:'):].strip()
    if final_answer.lower().startswith('a:'):
        final_answer = final_answer[len('a:'):].strip()

    # Strip wrapping angle brackets if the model wrapped the output (e.g. <place the orange in the bowl> or <yes>)
    while final_answer.startswith('<') and final_answer.endswith('>'):
        final_answer = final_answer[1:-1].strip()

    # Also remove any remaining single `<` or `>` characters
    final_answer = final_answer.replace('<', '').replace('>', '').strip()

    return final_answer


def evaluate_model(
    model_path: str,
    tfrecord_pattern: str,
    num_examples: int,
    device: str,
    max_new_tokens: int,
    batch_size: int,
    base_model_path: str = None,
    default_prompt: bool = True,
    max_num_frames: int = 6,
    enable_thinking: bool = False,
    repetition_penalty: float = None,
) -> None:
    """Loads VLA model, processes validation TFRecords, runs inference in batches, and prints stats."""
    if repetition_penalty is None:
        repetition_penalty = 1.0 if enable_thinking else 1.2
    print(f"[*] Dynamic repetition penalty configured: {repetition_penalty}")
    
    print(f"[*] Loading model and processor...")
    try:
        import json
        is_lora = False
        adapter_path = None
        base_path = base_model_path

        # Determine if model_path is a LoRA adapter directory or contains one
        if os.path.isdir(model_path):
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                is_lora = True
                adapter_path = model_path
            elif os.path.exists(os.path.join(model_path, "teacher_lora", "adapter_config.json")):
                is_lora = True
                adapter_path = os.path.join(model_path, "teacher_lora")

        if is_lora:
            print(f"[*] Detected LoRA adapter at: {adapter_path}")
            # Try to read base_model_name_or_path from adapter_config.json
            config_file = os.path.join(adapter_path, "adapter_config.json")
            try:
                with open(config_file, "r") as f:
                    adapter_config = json.load(f)
                config_base = adapter_config.get("base_model_name_or_path")
                if config_base and not base_path:
                    base_path = config_base
            except Exception as ce:
                print(f"[!] Warning: Could not read adapter config: {ce}")
            
            if not base_path:
                base_path = "shreethar/stage1_unsloth"
                
            print(f"[*] Loading base model and processor from: {base_path} ...")
            processor = AutoProcessor.from_pretrained(base_path)
            processor.video_processor.do_sample_frames = False
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_path,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map=device
            )
            
            from peft import PeftModel
            print(f"[*] Loading LoRA adapter from: {adapter_path} ...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print(f"[*] Loading full model and processor from: {model_path} ...")
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map=device
            )
    except Exception as e:
        print(f"[!] Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Configure padding side to left for batched generation
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"

    print(f"[*] Locating evaluation files using pattern: {tfrecord_pattern} ...")
    eval_filepaths = tf.io.gfile.glob(tfrecord_pattern)
    if not eval_filepaths:
        print(f"[!] No TFRecord files found matching pattern: {tfrecord_pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Found {len(eval_filepaths)} TFRecord file(s). Creating dataset ...")
    dataset = tf.data.TFRecordDataset(eval_filepaths)
    iterator = dataset.as_numpy_iterator()

    if num_examples <= 0:
        print(f"[*] Starting batched evaluation (batch size: {batch_size}) on all validation QA pairs ...\n")
        pbar = tqdm(desc="Evaluating")
    else:
        print(f"[*] Starting batched evaluation (batch size: {batch_size}) on up to {num_examples} QA pairs ...\n")
        pbar = tqdm(total=num_examples, desc="Evaluating")
    
    example_count = 0
    total_bleu = 0.0
    evaluated_pairs = []

    batch_messages = []
    batch_references = []
    batch_questions = []
    batch_task_types = []

    def flush_batch():
        nonlocal total_bleu
        if not batch_messages:
            return

        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            enable_thinking=enable_thinking,
        )

        # Move inputs to correct device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)

        input_ids_len = inputs['input_ids'].shape[1]

        # Run inference
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                stop_strings=["<|im_end|>"],
                eos_token_id=processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                repetition_penalty=repetition_penalty,
                tokenizer=processor.tokenizer,
                do_sample=True,
                temperature=0.1,
            )

        # Decode and evaluate batch items
        for idx in range(len(batch_messages)):
            generated_tokens = output_ids[idx, input_ids_len:]
            pred_raw = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract and clean prediction and reference answers
            pred_answer = clean_prediction(pred_raw)
            reference_answer = batch_references[idx]

            # Calculate BLEU score (case-insensitive to avoid penalizing case discrepancies)
            bleu_result = sacrebleu.sentence_bleu(pred_answer, [reference_answer], lowercase=True)
            bleu_score = bleu_result.score
            total_bleu += bleu_score

            evaluated_pairs.append({
                "index": example_count + idx + 1,
                "task_type": batch_task_types[idx],
                "question": batch_questions[idx],
                "reference": reference_answer,
                "pred_raw": pred_raw.strip(),
                "pred_cleaned": pred_answer,
                "bleu": bleu_score
            })
    
    try:
        for raw_record in iterator:
            if num_examples > 0 and example_count >= num_examples:
                break

            example = tf.train.SequenceExample()
            example.ParseFromString(raw_record)

            # Extract video frames (JPEG)
            images = []
            for bl in example.feature_lists.feature_list.get('images').feature:
                code = bl.bytes_list.value[0]
                image = tf.image.decode_jpeg(code).numpy()
                images.append(image)

            # Subsample frames to prevent context window bloat and speed up generation
            if len(images) > max_num_frames:
                step = len(images) / max_num_frames
                images = [images[int(i * step)] for i in range(max_num_frames)]

            # Get raw VQA script text and extract question-answer pairs
            texts_feature = example.feature_lists.feature_list.get("texts")
            if not texts_feature or not texts_feature.feature:
                continue

            raw_text = texts_feature.feature[0].bytes_list.value[0].decode('utf-8')
            qa_list = fetch_question_answer(raw_text)

            for _, task_type, question, answer in qa_list:
                if default_prompt:
                    if enable_thinking:
                        prompt_text = f"{STAGE2_QA_SYSTEM}\n\n{question}"
                    else:
                        prompt_text = question
                else:
                    prompt_text = (
                        "Output your final answer concisely, you may reason, but the final output should contain "
                        "</think> yes/no/place the paper on the table <|im_end|> etc, no need to give explanation on final answer. "
                        "Do not wrap your final answer in any brackets or tags (like < >). "
                        f"output within 10 words\nTask Instruction: {question}"
                    )

                batch_messages.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": images},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ])
                batch_references.append(answer.replace('A:', '').strip())
                batch_questions.append(question)
                batch_task_types.append(task_type)

                # Flush batch if it's full or if we hit the evaluation target limit
                if len(batch_messages) == batch_size or (num_examples > 0 and (example_count + len(batch_messages)) >= num_examples):
                    flush_batch()
                    example_count += len(batch_messages)
                    pbar.update(len(batch_messages))
                    
                    batch_messages.clear()
                    batch_references.clear()
                    batch_questions.clear()
                    batch_task_types.clear()

                    if num_examples > 0 and example_count >= num_examples:
                        break

            if num_examples > 0 and example_count >= num_examples:
                break

        # Flush any remaining items in the last batch
        if batch_messages:
            flush_batch()
            example_count += len(batch_messages)
            pbar.update(len(batch_messages))

    except KeyboardInterrupt:
        print("\n[!] Evaluation interrupted by user.")
    finally:
        pbar.close()

    if example_count == 0:
        print("[!] No evaluation pairs were processed.")
        return

    # Print Detailed Examples
    print("\n" + "="*80)
    print(" EVALUATION RESULTS (SAMPLES)")
    print("="*80)
    for pair in evaluated_pairs[:10]:  # Show up to first 10 examples
        print(f"\n--- Example {pair['index']} [{pair['task_type']}] ---")
        print(f"Reference: {pair['reference']}")
        print(f"Cleaned:   {pair['pred_cleaned']}")
        print(f"BLEU:      {pair['bleu']:.2f}%")

    # Print Final Summary
    mean_bleu = total_bleu / example_count
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    print(f"Total QA Pairs Evaluated: {example_count}")
    print(f"Mean BLEU Score:          {mean_bleu:.2f}%")
    print("="*80 + "\n")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLA checkpoint on RoboVQA Validation dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or HuggingFace hub ID of the model checkpoint to evaluate."
    )
    parser.add_argument(
        "--tfrecord_pattern",
        type=str,
        default="gs://gdm-robovqa/tfrecord/val/val*",
        help="Glob pattern matching the RoboVQA validation TFRecords."
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of question-answer pairs to evaluate. Set to -1 or 0 to evaluate all validation pairs."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Maximum new tokens generated by the model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for model inference."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path or HuggingFace hub ID of the base model if evaluating a LoRA adapter."
    )
    parser.add_argument(
        "--default_prompt",
        type=lambda x: str(x).lower() not in ("false", "0", "no"),
        default=True,
        help="Whether to use the default prompt from TFRecord (default: True). If False, a custom prompt format is used."
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=6,
        help="Maximum number of video frames to retain per record via uniform subsampling (default: 6)."
    )
    parser.add_argument(
        "--enable_thinking",
        type=lambda x: str(x).lower() not in ("false", "0", "no"),
        default=False,
        help="Whether to enable thinking mode during processor template application (default: False)."
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Repetition penalty for generation. Defaults to 1.0 if enable_thinking is True, else 1.2."
    )

    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        tfrecord_pattern=args.tfrecord_pattern,
        num_examples=args.num_examples,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        base_model_path=args.base_model_path,
        default_prompt=args.default_prompt,
        max_num_frames=args.max_num_frames,
        enable_thinking=args.enable_thinking,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()
