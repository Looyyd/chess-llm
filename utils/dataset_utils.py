#!/usr/bin/env python
# coding: utf-8

import random
import chess
from datasets import load_dataset
from .chess_utils import board_to_grid


def load_lichess_dataset(data_file, split="train", streaming=True, take=None):
    """Load lichess dataset with optional limiting"""
    dataset = load_dataset(
        "json",
        data_files=data_file,
        split=split,
        streaming=streaming,
    )
    
    if take is not None:
        dataset = dataset.take(take)
    
    return dataset


def select_weighted_position(moves):
    """
    Select a position from a game with weights that linearly increase from 1 to 5.
    Returns the position index.
    """
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
    return position_idx


def reconstruct_board_position(moves, position_idx):
    """
    Reconstruct board state up to a given position.
    Returns board, move_history list, and move_history string.
    """
    board = chess.Board()
    move_history = []
    
    for j in range(position_idx):
        move = moves[j]
        board.push_uci(move)
        move_history.append(move)
    
    # Format move history
    move_history_str = " ".join(move_history) if move_history else "Game start"
    
    return board, move_history, move_history_str


def get_turn_and_elo(board, white_elo, black_elo):
    """Get current turn and corresponding Elo rating"""
    if board.turn == chess.WHITE:
        current_elo = white_elo
        turn = "White"
    else:
        current_elo = black_elo
        turn = "Black"
    
    return turn, current_elo