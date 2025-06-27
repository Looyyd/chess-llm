#!/usr/bin/env python
# coding: utf-8

import chess
import chess.engine

# Configuration
STOCKFISH_PATH = r"C:\Users\filip\dev\stockfish\stockfish-windows-x86-64-avx2.exe"
# STOCKFISH_PATH = r"/usr/games/stockfish"
STOCKFISH_TIME_LIMIT = 0.1
STOCKFISH_DEPTH = 15

# Initialize Stockfish
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def test_reward_calculation():
    """Test the reward calculation with known positions"""
    
    # Test 1: White making a good move (e4)
    print("Test 1: White's good opening move e2e4")
    board = chess.Board()
    
    # Evaluate before
    info_before = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT, depth=STOCKFISH_DEPTH))
    eval_before = info_before["score"].relative.score()
    print(f"  Eval before (White to move): {eval_before}")
    
    # Make move
    board.push_san("e4")
    
    # Evaluate after
    info_after = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT, depth=STOCKFISH_DEPTH))
    eval_after = info_after["score"].relative.score()
    print(f"  Eval after (Black to move): {eval_after}")
    
    # Calculate reward
    reward = (-eval_after - eval_before) / 100
    print(f"  Reward for White: {reward:.2f} pawns")
    print(f"  Expected: Positive (good opening move)\n")
    
    # Test 2: Black making a good response (e5)
    print("Test 2: Black's good response e7e5")
    eval_before = eval_after  # Previous eval_after is now eval_before
    
    # Make move
    board.push_san("e5")
    
    # Evaluate after
    info_after = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT, depth=STOCKFISH_DEPTH))
    eval_after = info_after["score"].relative.score()
    print(f"  Eval before (Black to move): {eval_before}")
    print(f"  Eval after (White to move): {eval_after}")
    
    # Calculate reward
    reward = (-eval_after - eval_before) / 100
    print(f"  Reward for Black: {reward:.2f} pawns")
    print(f"  Expected: Positive (equalizing response)\n")
    
    # Test 3: White making a blunder (Qh5??)
    print("Test 3: White's blunder Qh5")
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    
    # Evaluate before
    info_before = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT, depth=STOCKFISH_DEPTH))
    eval_before = info_before["score"].relative.score()
    
    # Make blunder
    board.push_san("Qh5")
    
    # Evaluate after
    info_after = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT, depth=STOCKFISH_DEPTH))
    eval_after = info_after["score"].relative.score()
    print(f"  Eval before (White to move): {eval_before}")
    print(f"  Eval after (Black to move): {eval_after}")
    
    # Calculate reward
    reward = (-eval_after - eval_before) / 100
    print(f"  Reward for White: {reward:.2f} pawns")
    print(f"  Expected: Negative (early queen sortie)\n")

if __name__ == "__main__":
    try:
        test_reward_calculation()
    finally:
        engine.quit()