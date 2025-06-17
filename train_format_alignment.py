#!/usr/bin/env python
# coding: utf-8

import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from accelerate import PartialState
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import logging
from utils.chess_utils import extract_move_from_completion, has_thinking_tags

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
    model_name = "./chess_sft_qwen"  # Output from train_sft.py

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

    # Preprocess the dataset
    logger.info("Preprocessing dataset...")
    train_dataset = dataset.map(
        prepare_format_dataset,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
    )

    # Filter out empty texts
    train_dataset = train_dataset.filter(lambda example: len(example["text"]) > 0)

    logger.info(f"Dataset size after filtering: {len(train_dataset)}")

    # Prepare test prompts for validation callback
    test_prompts = []

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

    # Create a few test positions
    test_position = """Current game position:

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

What is the best move? Analyze the position and provide your answer."""

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

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,  # Smaller batch size for format training
        gradient_accumulation_steps=2,
        num_train_epochs=1,  # Usually one epoch is enough for format alignment
        max_steps=args.max_steps,
        learning_rate=5e-6,  # Lower learning rate to preserve capabilities
        warmup_steps=50,
        logging_steps=50,
        save_steps=250,
        save_total_limit=2,
        save_strategy="steps",
        eval_strategy="no",
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
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

    # Create the validation callback
    format_callback = FormatValidationCallback(
        tokenizer=tokenizer,
        test_prompts=test_prompts,
        inference_steps=100,  # Check more frequently
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[format_callback],
    )

    # Fine-tune the model
    logger.info("Starting format alignment training...")
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
        max_length=1024,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
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


if __name__ == "__main__":
    main()
