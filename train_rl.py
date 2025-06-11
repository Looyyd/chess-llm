#!/usr/bin/env python
# coding: utf-8

import torch
import random
import chess
import chess.engine
import re
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
import logging
from typing import List, Dict, Any
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEBUG = True
#STOCKFISH_PATH = r"C:\Users\filip\dev\stockfish\stockfish-windows-x86-64-avx2.exe"
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


def board_to_grid(board):
    """Convert board to visual grid representation"""
    grid_lines = []
    grid_lines.append("  a b c d e f g h")
    grid_lines.append("  ----------------")

    for rank in range(7, -1, -1):  # 8 to 1
        line = f"{rank + 1}|"
        for file in range(8):  # a to h
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is None:
                line += " ."
            else:
                symbol = piece.symbol()
                # Use uppercase for white, lowercase for black
                line += f" {symbol}"
        line += f" |{rank + 1}"
        grid_lines.append(line)

    grid_lines.append("  ----------------")
    grid_lines.append("  a b c d e f g h")

    return "\n".join(grid_lines)


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


def extract_move_from_completion(completion):
    """Extract move from the completion with thinking format"""
    # Look for move in \boxed{} format
    match = re.search(r"\\boxed\{([^}]+)\}", completion)
    if match:
        move_str = match.group(1).strip()
        return move_str
    return None


def has_thinking_tags(completion: str) -> bool:
    """Check if the completion contains <think> tags with content"""
    # Check for opening and closing tags
    think_pattern = r"<think>\s*(.+?)\s*</think>"
    match = re.search(think_pattern, completion, re.DOTALL)

    if match:
        # Ensure there's actual content between the tags
        content = match.group(1).strip()
        return len(content) > 0
    return False


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


def parse_board_from_prompt(prompt):
    """Parse chess board from the prompt text"""
    try:
        # Extract move history
        move_history_match = re.search(r"Move history \(UCI format\): ([^\n]+)", prompt)
        if move_history_match:
            move_history_str = move_history_match.group(1).strip()

            # Create a new board and apply moves
            board = chess.Board()

            if move_history_str != "Game start":
                moves = move_history_str.split()
                for move_str in moves:
                    try:
                        move = chess.Move.from_uci(move_str)
                        board.push(move)
                    except:
                        logger.warning(f"Failed to parse move: {move_str}")
                        return None

            return board

        return None
    except Exception as e:
        logger.error(f"Error parsing board from prompt: {e}")
        return None


def prepare_chess_dataset(examples, tokenizer):
    """Prepare chess games for GRPO training"""
    prompts = []

    for i in range(len(examples["moves"])):
        moves = examples["moves"][i]

        # Skip if game is too short
        if len(moves) < 2:
            continue

        # Create weights that linearly increase from 1 to 5
        # position_idx can be from 0 to len(moves)-2 (inclusive)
        num_positions = len(moves) - 1

        # Create weights: weight = 1 + 4 * (position / (num_positions - 1))
        # This gives us 1 for position 0 and 5 for the last valid position
        weights = []
        for pos in range(num_positions):
            if num_positions == 1:
                weight = 1.0  # Only one position available
            else:
                weight = 1.0 + 4.0 * (pos / (num_positions - 1))
            weights.append(weight)

        # Use random.choices with weights to select position
        position_idx = random.choices(range(num_positions), weights=weights, k=1)[0]

        # Reconstruct board up to that position
        board = chess.Board()
        move_history = []

        for j in range(position_idx):
            move = moves[j]
            board.push_uci(move)
            move_history.append(move)

        # Determine whose turn it is
        turn = "White" if board.turn == chess.WHITE else "Black"

        # Format move history
        move_history_str = " ".join(move_history) if move_history else "Game start"

        # Create board visualization
        board_grid = board_to_grid(board)

        # Create the prompt with thinking format instruction
        system_prompt = """You are a chess engine. Given a chess position, analyze the position and determine the best move.

First, analyze the position inside <think> tags, considering:
- Material balance
- Piece activity and coordination
- King safety
- Pawn structure
- Tactical opportunities
- Strategic plans

Then provide your chosen move in UCI format inside \\boxed{} tags.

Example format:
<think>
The position shows an open center with both sides castled kingside. White has a slight space advantage...
The knight on f3 can jump to e5, attacking the weak f7 square...
</think>
\\boxed{e4e5}"""

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
    base_model_path = "./chess_lora_qwen"  # Path to your fine-tuned model

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

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files="./data/lichess_2013_12_compact.jsonl",
        split="train",
        streaming=True,
    )

    if DEBUG:
        # Take only first 100 games for debugging
        dataset = dataset.take(100)

    # Preprocess the dataset
    logger.info("Preprocessing dataset...")
    train_dataset = dataset.map(
        lambda examples: prepare_chess_dataset(examples, tokenizer),
        batched=True,
        batch_size=10,
        remove_columns=dataset.column_names,
    )

    # Filter out empty prompts
    train_dataset = train_dataset.filter(lambda example: len(example["prompt"]) > 0)

    # GRPO configuration
    training_args = GRPOConfig(
        output_dir="./chess_grpo_qwen",
        # Training parameters
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=500 if DEBUG else -1,
        # Generation parameters
        num_generations=4,  # Number of completions to generate per prompt
        max_prompt_length=1024,
        max_completion_length=512,  # Longer to accommodate thinking
        temperature=0.8,
        top_p=0.95,
        # GRPO specific parameters
        beta=0.04,  # No KL penalty (following recent best practices)
        epsilon=0.2,  # Clipping parameter
        epsilon_high=0.28,
        reward_weights=None,  # Single reward function
        scale_rewards=False,
        loss_type="dr_grpo",  # Use Dr. GRPO to avoid length bias
        mask_truncated_completions=True,
        # Optimization
        learning_rate=5e-6,
        warmup_steps=50,
        optim="adamw_8bit",
        max_grad_norm=1.0,
        # Logging and saving
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        save_strategy="steps",
        push_to_hub=not DEBUG,
        log_completions=True,
        num_completions_to_print=2,
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
        vllm_mode="server",
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=base_model_path,
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
        {"role": "system", "content": training_args.system_prompt},
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
