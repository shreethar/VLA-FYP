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

import tensorflow as tf
import torch
import sacrebleu
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from absl import logging

# Disable tensorflow logs to keep output clean unless needed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.set_verbosity(logging.ERROR)

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

    return final_answer


def evaluate_model(
    model_path: str,
    tfrecord_pattern: str,
    num_examples: int,
    device: str,
    max_new_tokens: int,
    batch_size: int,
) -> None:
    """Loads VLA model, processes validation TFRecords, runs inference in batches, and prints stats."""
    
    print(f"[*] Loading model and processor from: {model_path} ...")
    try:
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
            enable_thinking=False,
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
                repetition_penalty=1.2,
                tokenizer=processor.tokenizer
            )

        # Decode and evaluate batch items
        for idx in range(len(batch_messages)):
            generated_tokens = output_ids[idx, input_ids_len:]
            pred_raw = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract and clean prediction and reference answers
            pred_answer = clean_prediction(pred_raw)
            reference_answer = batch_references[idx]

            # Calculate BLEU score
            bleu_result = sacrebleu.sentence_bleu(pred_answer, [reference_answer])
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

            # Get raw VQA script text and extract question-answer pairs
            texts_feature = example.feature_lists.feature_list.get("texts")
            if not texts_feature or not texts_feature.feature:
                continue

            raw_text = texts_feature.feature[0].bytes_list.value[0].decode('utf-8')
            qa_list = fetch_question_answer(raw_text)

            for _, task_type, question, answer in qa_list:
                batch_messages.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": images},
                            {"type": "text", "text": question}
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
        print(f"Question:  {pair['question']}")
        print(f"Reference: {pair['reference']}")
        print(f"Raw Pred:  {pair['pred_raw']}")
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

    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        tfrecord_pattern=args.tfrecord_pattern,
        num_examples=args.num_examples,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
