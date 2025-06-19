#!/usr/bin/env python
# coding: utf-8

import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from accelerate import PartialState
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import logging
from utils.chess_utils import extract_move_from_completion, has_thinking_tags
from utils.prompt_utils import get_chess_system_prompt
import numpy as np

# In case previous experiments didn't close properly
torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False


class FormatValidationCallback(TrainerCallback):
    """Callback to validate format during training"""

    def __init__(self, tokenizer, test_prompts, inference_steps=500):
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.inference_steps = inference_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Run inference every `inference_steps` steps
        if state.global_step % self.inference_steps == 0 and state.global_step > 0:
            model = kwargs["model"]

            logger.info(f"\n{'='*50}")
            logger.info(f"Running format validation at step {state.global_step}")
            logger.info(f"{'='*50}")

            # Test on a random prompt
            import random

            test_prompt = random.choice(self.test_prompts)

            # Prepare the input
            inputs = self.tokenizer(
                test_prompt,
                return_tensors="pt",
            ).to(model.device)

            # Generate
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,  # Longer for thinking
                    temperature=0.7,
                    do_sample=True,
                )

            # Decode and print
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response in format compliance: {response}")
            # Only show the assistant response part
            if "<|im_start|>assistant" in response:
                assistant_part = response.split("<|im_start|>assistant")[-1].strip()
            else:
                assistant_part = response

            logger.info(
                f"Model response:\n{assistant_part[:500]}..."
            )  # First 500 chars

            # Check format compliance
            has_think = has_thinking_tags(assistant_part)
            predicted_move = extract_move_from_completion(assistant_part)

            logger.info(f"Has thinking tags: {has_think}")
            logger.info(f"Extracted move: {predicted_move}")

            logger.info(f"{'='*50}\n")

            model.train()


class FormatComplianceMetricsCallback(TrainerCallback):
    """Callback to compute format compliance metrics during evaluation"""

    def __init__(self, tokenizer, eval_prompts):
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts
        self.format_compliance_scores = []

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]

        logger.info(f"\n{'='*50}")
        logger.info("Computing format compliance metrics on evaluation set")
        logger.info(f"{'='*50}")

        has_thinking_count = 0
        has_move_count = 0
        total_samples = min(10, len(self.eval_prompts))  # Evaluate on up to 10 samples

        model.eval()
        for i in range(total_samples):
            test_prompt = self.eval_prompts[i % len(self.eval_prompts)]

            inputs = self.tokenizer(
                test_prompt,
                return_tensors="pt",
            ).to(model.device)

            logger.info(f"Format Compliance INPUTS")
            logger.info(inputs)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(outputs[-1], skip_special_tokens=True)
            logger.info(f"Format Compliance Response")
            logger.info(response)

            if "<|im_start|>assistant" in response:
                assistant_part = response.split("<|im_start|>assistant")[-1].strip()
            else:
                assistant_part = response

            # Check format compliance
            has_think = has_thinking_tags(assistant_part)
            predicted_move = extract_move_from_completion(assistant_part)

            if has_think:
                has_thinking_count += 1
            if predicted_move:
                has_move_count += 1

        thinking_rate = has_thinking_count / total_samples
        move_rate = has_move_count / total_samples

        logger.info(f"Format Compliance Metrics:")
        logger.info(
            f"  Thinking tags rate: {thinking_rate:.2%} ({has_thinking_count}/{total_samples})"
        )
        logger.info(
            f"  Valid move rate: {move_rate:.2%} ({has_move_count}/{total_samples})"
        )
        logger.info(f"{'='*50}\n")

        # Store for tracking
        self.format_compliance_scores.append(
            {
                "step": state.global_step,
                "thinking_rate": thinking_rate,
                "move_rate": move_rate,
            }
        )

        model.train()

        # Add metrics to the evaluation results
        if "metrics" in kwargs:
            kwargs["metrics"]["eval_thinking_rate"] = thinking_rate
            kwargs["metrics"]["eval_move_rate"] = move_rate

        return control


def prepare_format_dataset(examples):
    """Prepare the format alignment dataset"""
    # The dataset should already be in the correct format with messages
    # We just need to extract the text field
    texts = []

    for messages in examples["messages"]:
        # Apply chat template to the messages
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)

    return {"text": texts}


def main():
    global tokenizer  # Make tokenizer global for the prepare function

    # Model configuration - using the model from the first SFT stage
    model_name = "./chess_sft_qwen_hf/checkpoint-5000/"  # Output from train_sft.py

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Format alignment training for chess model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Looyyd/chess-format-alignment",
        help="Hugging Face dataset path",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Maximum training steps"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./chess_format_aligned",
        help="Output directory for the model",
    )
    parser.add_argument(
        "--eval-split-ratio",
        type=float,
        default=0.05,
        help="Ratio of data to use for evaluation (default: 0.05)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=20,
        help="Run evaluation every N steps (default: 20)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Number of evaluation steps with no improvement before early stopping (default: 3)",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.001,
        help="Minimum change in loss to qualify as improvement (default: 0.001)",
    )
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Load dataset from Hugging Face
    logger.info(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(
        args.dataset,
        data_files={"train": "train.jsonl"},
        split="train",
    )

    # If dataset is large, optionally limit it
    if DEBUG:
        dataset = dataset.select(range(min(100, len(dataset))))

    # Split the dataset into train and eval
    logger.info(f"Splitting dataset with eval ratio: {args.eval_split_ratio}")
    split_dataset = dataset.train_test_split(
        test_size=args.eval_split_ratio, seed=42, shuffle=True  # For reproducibility
    )

    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Prepare test prompts for validation callback
    test_prompts = []
    eval_prompts = []

    # Get system prompt from shared utility
    system_prompt = get_chess_system_prompt()

    # Training arguments with evaluation
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=16,  # Smaller batch size for format training
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,  # Usually one epoch is enough for format alignment
        max_steps=args.max_steps,
        learning_rate=1e-6,  # Lower learning rate to preserve capabilities
        warmup_steps=50,
        logging_steps=10,
        save_steps=args.eval_steps,
        save_total_limit=2,
        save_strategy="steps",
        # eval_strategy="steps",  # Enable evaluation
        # eval_steps=args.eval_steps,  # Evaluate every N steps
        # metric_for_best_model="eval_loss",  # Use eval loss for model selection
        # greater_is_better=False,  # Lower loss is better
        # load_best_model_at_end=True,  # Load best model when training ends
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        # Completion only, to train on what we really care about
        completion_only_loss=True,
        # SFT specific parameters
        max_length=4096,
        packing=False,
        # dataset_text_field="text",
        report_to="none" if DEBUG else "wandb",
        push_to_hub=not DEBUG,
        hub_strategy="end",
        accelerator_config={
            "dispatch_batches": False,
        },
        use_liger_kernel=True,
        # Gradient checkpointing for memory efficiency
        gradient_checkpointing=True,
    )

    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )

    # Initialize trainer with evaluation dataset
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,  # Add evaluation dataset
        processing_class=tokenizer,
    )

    # Fine-tune the model
    logger.info("Starting format alignment training with evaluation...")
    trainer.train()

    # Save the model
    trainer.save_model()
    logger.info(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
