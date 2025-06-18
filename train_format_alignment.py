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
import numpy as np

# In case previous experiments didn't close properly
torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False


class FormatValidationCallback(TrainerCallback):
    """Callback to validate format during training"""

    def __init__(self, tokenizer, test_prompts, inference_steps=500):
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
                truncation=True,
                max_length=1024,
            ).to(model.device)

            # Generate
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,  # Longer for thinking
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode and print
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
                truncation=True,
                max_length=1024,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

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

    # Load checklist for test prompt
    def load_checklist(phase):
        checklist_path = os.path.join(
            os.path.dirname(__file__), "checklists", f"{phase}.md"
        )
        try:
            with open(checklist_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Checklist file not found: {checklist_path}")
            return ""

    ALL_CHECKLISTS = f"""
{load_checklist("opening")}

---

{load_checklist("midgame")}

---

{load_checklist("endgame")}
"""

    system_prompt = f"""You are a chess engine. Given a chess position, analyze the position and determine the best move.

First, analyze the position inside <think> tags, using the following checklists to guide your thinking:

{ALL_CHECKLISTS}

Choose the most appropriate checklist(s) based on the game phase and work through them systematically as you analyze the position. Then provide your chosen move in UCI format inside \\boxed{{}} tags.

Example format:
<think>
The position appears to be in the [opening/middlegame/endgame] phase. Following the relevant checklist:

1. Safety & Basic Tactics:
- My king is safe on g1, not in check
- No pieces are hanging
- No immediate captures available...

2. [Continue through the relevant checklist sections...]

Based on this analysis, the best move is...
</think>
\\boxed{{f3e5}}"""

    # Create test positions for validation and evaluation
    test_positions = [
        """Current game position:

Move history (UCI format): e2e4 e7e5 g1f3 b8c6 f1c4 g8f6
Turn: White

Current board state:
  a b c d e f g h
  ----------------
8| r . b q k b . r |8
7| p p p p . p p p |7
6| . . n . . n . . |6
5| . . . . p . . . |5
4| . . B . P . . . |4
3| . . . . . N . . |3
2| P P P P . P P P |2
1| R N B Q K . . R |1
  ----------------
  a b c d e f g h

What is the best move? Analyze the position and provide your answer.""",
        """Current game position:

Move history (UCI format): d2d4 g8f6 c2c4 e7e6 g1f3 d7d5
Turn: White

Current board state:
  a b c d e f g h
  ----------------
8| r n b q k b . r |8
7| p p p . . p p p |7
6| . . . . p n . . |6
5| . . . p . . . . |5
4| . . P P . . . . |4
3| . . . . . N . . |3
2| P P . . P P P P |2
1| R N B Q K B . R |1
  ----------------
  a b c d e f g h

What is the best move? Analyze the position and provide your answer.""",
    ]

    # Create prompts for both validation and evaluation
    for test_position in test_positions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_position},
        ]

        # Apply chat template and add <think> prefix
        test_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        test_prompt += "<think>"
        test_prompts.append(test_prompt)
        eval_prompts.append(test_prompt)

    # Training arguments with evaluation
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=16,  # Smaller batch size for format training
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,  # Usually one epoch is enough for format alignment
        max_steps=args.max_steps,
        learning_rate=5e-6,  # Lower learning rate to preserve capabilities
        warmup_steps=50,
        logging_steps=10,
        save_steps=args.eval_steps,
        save_total_limit=2,
        save_strategy="steps",
        eval_strategy="steps",  # Enable evaluation
        eval_steps=args.eval_steps,  # Evaluate every N steps
        metric_for_best_model="eval_loss",  # Use eval loss for model selection
        greater_is_better=False,  # Lower loss is better
        load_best_model_at_end=True,  # Load best model when training ends
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        # Completion only, to train on what we really care about
        completion_only_loss=True,
        # SFT specific parameters
        max_length=1024,
        packing=False,
        dataset_text_field="text",
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

    # Create callbacks
    format_validation_callback = FormatValidationCallback(
        tokenizer=tokenizer,
        test_prompts=test_prompts,
        inference_steps=100,  # Check more frequently
    )

    format_metrics_callback = FormatComplianceMetricsCallback(
        tokenizer=tokenizer,
        eval_prompts=eval_prompts,
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
        eval_dataset=eval_dataset,  # Add evaluation dataset
        processing_class=tokenizer,
        callbacks=[
            format_validation_callback,
            format_metrics_callback,
            early_stopping_callback,
        ],
    )

    # Fine-tune the model
    logger.info("Starting format alignment training with evaluation...")
    trainer.train()

    # Save the model
    trainer.save_model()
    logger.info(f"Training complete! Model saved to {args.output_dir}")

    # Final validation test
    logger.info("\nFinal format validation:")
    inputs = tokenizer(
        test_prompts[0],
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in response:
        assistant_part = response.split("<|im_start|>assistant")[-1].strip()
    else:
        assistant_part = response

    print(f"Model response:\n{assistant_part}")

    # Check format
    has_think = has_thinking_tags(assistant_part)
    predicted_move = extract_move_from_completion(assistant_part)

    print(f"\nFormat check:")
    print(f"Has thinking tags: {has_think}")
    print(f"Extracted move: {predicted_move}")

    # Print final format compliance scores
    if (
        hasattr(format_metrics_callback, "format_compliance_scores")
        and format_metrics_callback.format_compliance_scores
    ):
        print("\nFormat compliance throughout training:")
        for score in format_metrics_callback.format_compliance_scores:
            print(
                f"Step {score['step']}: Thinking rate={score['thinking_rate']:.2%}, Move rate={score['move_rate']:.2%}"
            )


if __name__ == "__main__":
    main()
