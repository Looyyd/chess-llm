#!/usr/bin/env python
# coding: utf-8

import torch
import random
import chess
import chess.engine
import re
import os
from transformers import AutoTokenizer
from datasets import load_dataset
import logging
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
from utils.prompt_utils import (
    get_chess_system_prompt,
    create_chess_user_prompt,
    format_chess_messages,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
STOCKFISH_PATH = r"C:\Users\filip\dev\stockfish\stockfish-windows-x86-64-avx2.exe"
STOCKFISH_TIME_LIMIT = 1  # Time limit for stockfish analysis in seconds
STOCKFISH_DEPTH = 20  # Depth for stockfish analysis

# Initialize Stockfish engine
try:
    print(f"\n[INIT] Attempting to initialize Stockfish at: {STOCKFISH_PATH}")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    print("[INIT] Stockfish initialized successfully!")
except Exception as e:
    print(f"[ERROR] Failed to initialize Stockfish: {e}")
    print("[ERROR] Please ensure Stockfish is installed at the specified path")
    raise

# Get the system prompt
system_prompt = get_chess_system_prompt()


def evaluate_position(board):
    """Evaluate a chess position using Stockfish with detailed logging"""
    try:
        print(f"\n[EVAL] Evaluating position...")
        print(f"[EVAL] FEN: {board.fen()}")
        print(f"[EVAL] Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        info = engine.analyse(
            board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT, depth=STOCKFISH_DEPTH)
        )
        
        # Get the score in centipawns
        score = info["score"].relative
        print(f"[EVAL] Raw score object: {score}")
        
        if score.is_mate():
            # Convert mate score to a large value
            mate_in = score.mate()
            print(f"[EVAL] Position is mate in {mate_in}")
            if mate_in > 0:
                result = 10000 - mate_in  # Positive for winning
                print(f"[EVAL] Winning mate score: {result}")
            else:
                result = -10000 - mate_in  # Negative for losing
                print(f"[EVAL] Losing mate score: {result}")
            return result
        else:
            # Return centipawns
            centipawns = score.score()
            print(f"[EVAL] Centipawn evaluation: {centipawns}")
            return centipawns
    except Exception as e:
        print(f"[ERROR] Error evaluating position: {e}")
        return 0


def calculate_reward_for_move(board, move_str):
    """Calculate the reward for a specific move with detailed logging"""
    print(f"\n{'='*60}")
    print(f"[REWARD] Calculating reward for move: {move_str}")
    print(f"{'='*60}")
    
    # Show current position
    print(f"\n[BOARD] Current position:")
    print(board_to_grid(board))
    
    base_reward = 0
    
    # Try to parse and apply the move
    try:
        print(f"\n[MOVE] Attempting to parse move: {move_str}")
        move = chess.Move.from_uci(move_str)
        print(f"[MOVE] Parsed move object: {move}")
        
        print(f"\n[MOVE] Checking if move is legal...")
        legal_moves = list(board.legal_moves)
        print(f"[MOVE] Legal moves: {[m.uci() for m in legal_moves[:10]]}{'...' if len(legal_moves) > 10 else ''}")
        
        if move not in board.legal_moves:
            print(f"[MOVE] ❌ ILLEGAL MOVE! {move_str} is not legal in this position")
            print(f"[REWARD] Final reward: {base_reward - 12.0} (base: {base_reward}, illegal move penalty: -12.0)")
            return base_reward - 12.0
        
        print(f"[MOVE] ✓ Move is legal!")
        
        # Evaluate position before move
        print(f"\n[BEFORE] Evaluating position BEFORE move...")
        eval_before = evaluate_position(board)
        
        # Apply the move
        print(f"\n[MOVE] Applying move {move_str} to board...")
        board_copy = board.copy()  # Make a copy to not modify original
        board_copy.push(move)
        
        # Evaluate position after move
        print(f"\n[AFTER] Evaluating position AFTER move...")
        eval_after = evaluate_position(board_copy)
        
        # Calculate reward as the change in evaluation
        print(f"\n[CALC] Calculating reward...")
        print(f"[CALC] Evaluation before move (from {'White' if board.turn == chess.WHITE else 'Black'}'s perspective): {eval_before}")
        print(f"[CALC] Evaluation after move (from {'White' if board_copy.turn == chess.WHITE else 'Black'}'s perspective): {eval_after}")
        print(f"[CALC] Player who made the move: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        # Since Stockfish's relative evaluation is always from the perspective of the side to move:
        # - eval_before is from the perspective of the player making the move
        # - eval_after is from the perspective of the opponent (after turn switches)
        # So we need to negate eval_after to get both from the same perspective
        print(f"\n[CALC] Stockfish evaluations are relative to side to move")
        print(f"[CALC] eval_before is from {'White' if board.turn == chess.WHITE else 'Black'}'s perspective: {eval_before}")
        print(f"[CALC] eval_after is from {'White' if board_copy.turn == chess.WHITE else 'Black'}'s perspective: {eval_after}")
        print(f"[CALC] To compare, we negate eval_after to get it from the mover's perspective: {-eval_after}")
        
        # The reward is how much the position improved for the player who made the move
        reward = (-eval_after - eval_before) / 100  # Centipawns to pawns
        print(f"[CALC] Raw reward: ({-eval_after} - {eval_before}) / 100 = {reward}")
        
        # Clip reward to reasonable range
        print(f"\n[CLIP] Clipping reward to range [-10, 10]...")
        print(f"[CLIP] Before clipping: {reward}")
        reward = max(-10.0, min(10.0, reward))
        print(f"[CLIP] After clipping: {reward}")
        
        final_reward = base_reward + float(reward)
        print(f"\n[REWARD] Final reward: {final_reward} (base: {base_reward}, move reward: {reward})")
        
        return final_reward
        
    except (ValueError, chess.InvalidMoveError) as e:
        print(f"[ERROR] Invalid move format {move_str}: {e}")
        print(f"[REWARD] Final reward: {base_reward - 12.0} (base: {base_reward}, invalid move penalty: -12.0)")
        return base_reward - 12.0


def main():
    # Load tokenizer for creating prompts
    print("\n[LOAD] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Looyyd/chess-format-aligned-qwen")
    
    # Load dataset
    print("[LOAD] Loading chess dataset...")
    dataset = load_dataset(
        "Looyyd/chess-dataset",
        data_files={"train": "train.jsonl"},
        split="train",
        streaming=True,
    )
    
    # Get a sample from the dataset
    print("\n[SAMPLE] Getting a sample game from the dataset...")
    sample = next(iter(dataset))
    moves = sample["moves"]
    
    print(f"[SAMPLE] Found game with {len(moves)} moves")
    print(f"[SAMPLE] First 10 moves: {moves[:10]}")
    
    # Select a position
    position_idx = select_weighted_position(moves)
    print(f"\n[POSITION] Selected position at move {position_idx}")
    
    # Reconstruct board position
    board, move_history, move_history_str = reconstruct_board_position(moves, position_idx)
    
    # Create prompt (for display)
    turn = "White" if board.turn == chess.WHITE else "Black"
    board_grid = board_to_grid(board)
    user_prompt = create_chess_user_prompt(board_grid, move_history_str, turn)
    
    print(f"\n[PROMPT] Generated prompt:")
    print("-" * 60)
    print(user_prompt)
    print("-" * 60)
    
    # Interactive loop
    while True:
        print(f"\n[INPUT] Enter a move in UCI format (e.g., e2e4, e7e8q) or 'quit' to exit:")
        print(f"[INPUT] Current turn: {turn}")
        move_input = input("> ").strip()
        
        if move_input.lower() == 'quit':
            break
        
        # Test the reward calculation
        reward = calculate_reward_for_move(board, move_input)
        
        print(f"\n{'='*60}")
        print(f"[SUMMARY] Move: {move_input}")
        print(f"[SUMMARY] Reward: {reward}")
        print(f"{'='*60}")
        
        # Ask if user wants to try another move
        print("\n[CONTINUE] Try another move with the same position? (y/n)")
        if input("> ").strip().lower() != 'y':
            # Get a new position
            print("\n[NEW] Getting a new position...")
            sample = next(iter(dataset))
            moves = sample["moves"]
            position_idx = select_weighted_position(moves)
            board, move_history, move_history_str = reconstruct_board_position(moves, position_idx)
            turn = "White" if board.turn == chess.WHITE else "Black"
            board_grid = board_to_grid(board)
            user_prompt = create_chess_user_prompt(board_grid, move_history_str, turn)
            
            print(f"\n[PROMPT] New position:")
            print("-" * 60)
            print(user_prompt)
            print("-" * 60)


def cleanup():
    """Clean up resources"""
    global engine
    if engine:
        print("\n[CLEANUP] Closing Stockfish engine...")
        engine.quit()
        print("[CLEANUP] Done!")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()