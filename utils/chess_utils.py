#!/usr/bin/env python
# coding: utf-8

import re
import chess


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
                        return None

            return board

        return None
    except Exception as e:
        return None


def parse_time_control(time_control):
    """Safely parse time control string"""
    if not time_control or time_control == "-":
        return None  # or 0, depending on how you want to handle it

    if "+" in time_control:
        try:
            base_time = int(time_control.split("+")[0])
            return base_time
        except (ValueError, IndexError):
            return None
    else:
        # Handle formats like "90" (just minutes)
        try:
            return int(time_control)
        except ValueError:
            return None