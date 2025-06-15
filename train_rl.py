#!/usr/bin/env python
# coding: utf-8

import torch
import random
import chess
import chess.engine
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import PartialState
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
import logging
import argparse
from utils.chess_utils import (
    board_to_grid,
    extract_move_from_completion,
    has_thinking_tags,
    parse_board_from_prompt,
)
from utils.dataset_utils import (
    select_weighted_position,
    reconstruct_board_position,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEBUG = False
# STOCKFISH_PATH = r"C:\Users\filip\dev\stockfish\stockfish-windows-x86-64-avx2.exe"
STOCKFISH_PATH = r"/usr/games/stockfish"

STOCKFISH_TIME_LIMIT = 0.1  # Time limit for stockfish analysis in seconds
STOCKFISH_DEPTH = 15  # Depth for stockfish analysis

# Initialize Stockfish engine globally to avoid repeated initialization
try:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
except Exception as e:
    logger.error(f"Failed to initialize Stockfish: {e}")
    logger.error("Please install Stockfish and update STOCKFISH_PATH in the script")
    raise


def load_checklist(phase):
    """Load checklist markdown file for a specific game phase"""
    checklist_path = os.path.join(
        os.path.dirname(__file__), "checklists", f"{phase}.md"
    )
    try:
        with open(checklist_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Checklist file not found: {checklist_path}")
        return ""


# Cache all checklists at module level to avoid repeated file I/O
ALL_CHECKLISTS = f"""
{load_checklist("opening")}

---

{load_checklist("midgame")}

---

{load_checklist("endgame")}
"""

# Create the prompt with thinking format instruction
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


def evaluate_position(board):
    """Evaluate a chess position using Stockfish"""
    try:
        info = engine.analyse(
            board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT, depth=STOCKFISH_DEPTH)
        )

        # Get the score in centipawns
        score = info["score"].relative

        if score.is_mate():
            # Convert mate score to a large value
            mate_in = score.mate()
            if mate_in > 0:
                return 10000 - mate_in  # Positive for winning
            else:
                return -10000 - mate_in  # Negative for losing
        else:
            # Return centipawns
            return score.score()
    except Exception as e:
        logger.error(f"Error evaluating position: {e}")
        return 0


def chess_reward_function(prompts, completions, **kwargs):
    """
    Reward function based on Stockfish evaluation difference.
    The reward is the change in evaluation after making the move.
    """
    rewards = []

    for prompt, completion in zip(prompts, completions):
        try:
            # Parse the board state from the prompt
            board = parse_board_from_prompt(prompt)
            if board is None:
                if DEBUG:
                    logger.warning("Failed to parse board from prompt")
                rewards.append(0.0)
                continue

            base_reward = 0
            if not has_thinking_tags(completion):
                base_reward -= 5  # no thinking tags penalty

            # Extract the move from the completion
            move_str = extract_move_from_completion(completion)
            if move_str is None:
                if DEBUG:
                    logger.warning(
                        f"Failed to extract move from completion: {completion[:100]}..."
                    )
                rewards.append(
                    base_reward - 15.0
                )  # Penalty for invalid format(needs to be greated than the move rewards, to avoid reward hacking)
                continue

            # Try to parse and apply the move
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    if DEBUG:
                        logger.warning(f"Illegal move {move_str} in position")
                    rewards.append(
                        base_reward - 12.0
                    )  # Heavy penalty for illegal moves
                    continue

                # Evaluate position before move
                eval_before = evaluate_position(board)

                # Apply the move
                board.push(move)

                # Evaluate position after move
                eval_after = evaluate_position(board)

                # Calculate reward as the change in evaluation
                # Note: We need to negate for black's moves
                if board.turn == chess.BLACK:  # After the move, turn switches
                    reward = (eval_after - eval_before) / 100  # Centipawns to pawns
                else:
                    reward = (eval_before - eval_after) / 100

                # Clip reward to reasonable range, and this ensures that play reward is always better than invalid format
                # To avoid reward hacking with invalid moves
                reward = max(-10.0, min(10.0, reward))

                rewards.append(base_reward + float(reward))

            except (ValueError, chess.InvalidMoveError) as e:
                if DEBUG:
                    logger.warning(f"Invalid move format {move_str}: {e}")
                rewards.append(base_reward - 12.0)  # Penalty for invalid move format

        except Exception as e:
            logger.error(f"Error in reward calculation: {e}")
            rewards.append(0.0)

    return rewards


def prepare_chess_dataset(examples, tokenizer):
    """Prepare chess games for GRPO training"""
    prompts = []

    for i in range(len(examples["moves"])):
        moves = examples["moves"][i]

        # Skip if game is too short
        # NOTE: here we don't filter as aggressively as in train_sft, because having lower quality games is good, we want model to perform in variety of positions
        if len(moves) < 8:
            continue

        # Select weighted position
        position_idx = select_weighted_position(moves)

        # Reconstruct board position
        board, move_history, move_history_str = reconstruct_board_position(
            moves, position_idx
        )

        # Determine whose turn it is
        turn = "White" if board.turn == chess.WHITE else "Black"

        # Create board visualization
        board_grid = board_to_grid(board)

        user_prompt = f"""Current game position:

Move history (UCI format): {move_history_str}
Turn: {turn}

Current board state:
{board_grid}

What is the best move? Analyze the position and provide your answer."""

        # Format as messages for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompts.append(prompt)

    return {"prompt": prompts}


def main():
    # Model configuration
    base_model_path = "./chess_sft_qwen"  # Path to your fine-tuned model

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train a chess reinforcement learning model."
    )
    parser.add_argument(
        "--vllm", action="store_true", help="Use vLLM for model generation."
    )
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for GRPO

    # Load dataset from Hugging Face
    logger.info("Loading dataset from Hugging Face...")
    take_count = 100 if DEBUG else 100_000

    # Load from Hugging Face Hub
    dataset = load_dataset(
        "Looyyd/chess-dataset",
        data_files={"train": "train.jsonl"},
        split="train",
        streaming=False,  # streaming not compatible with GRPOTrainer just yet
    )

    if take_count is not None:
        dataset = dataset.take(take_count)

    # Preprocess the dataset
    logger.info("Preprocessing dataset...")
    train_dataset = dataset.map(
        lambda examples: prepare_chess_dataset(examples, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
    )

    # Filter out empty prompts
    train_dataset = train_dataset.filter(lambda example: len(example["prompt"]) > 0)

    # GRPO configuration
    training_args = GRPOConfig(
        output_dir="./chess_grpo_qwen",
        # Training parameters
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=500 if DEBUG else 50_000,
        gradient_checkpointing=True,
        # Generation parameters
        num_generations=8,
        max_prompt_length=1024,
        max_completion_length=512,  # Longer to accommodate thinking
        temperature=0.8,
        top_p=0.95,
        # GRPO specific parameters
        beta=0.04,
        epsilon=0.2,  # Clipping parameter
        epsilon_high=0.28,
        reward_weights=None,  # Single reward function
        scale_rewards=False,
        loss_type="dr_grpo",  # Use Dr. GRPO to avoid length bias
        mask_truncated_completions=True,
        # Optimization
        learning_rate=1e-6,
        warmup_steps=50,
        max_grad_norm=1.0,
        # Logging and saving
        logging_steps=1,
        save_steps=100,
        save_total_limit=1,
        save_strategy="steps",
        push_to_hub=not DEBUG,
        log_completions=True,
        log_on_each_node=False,
        num_completions_to_print=1,
        sync_ref_model=True,  # Probably a good idea for long runs, recommended in https://github.com/willccbb/verifiers
        ref_model_sync_steps=512,  # Default value
        # Other settings
        bf16=True,
        report_to="none" if DEBUG else "wandb",
        seed=42,
        accelerator_config={
            # Otherwise the variable length sequences can cause issues on multi gpu
            "dispatch_batches": False,
        },
        # VLLM
        use_vllm=args.vllm,
        use_liger_kernel=True,
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=chess_reward_function,
    )

    # Train the model
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save the final model
    trainer.save_model()
    logger.info("Training complete! Model saved.")

    # Test the trained model
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

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to("cuda")

    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model response:\n{response}")

    # Extract and display the move
    move = extract_move_from_completion(response)
    if move:
        print(f"\nPredicted move: {move}")


def cleanup():
    """Clean up resources"""
    global engine
    if engine:
        engine.quit()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
