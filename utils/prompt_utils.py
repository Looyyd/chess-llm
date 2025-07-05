#!/usr/bin/env python
# coding: utf-8

import os
import logging

logger = logging.getLogger(__name__)


def load_checklist(phase):
    """Load checklist markdown file for a specific game phase"""
    checklist_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "checklists", f"{phase}.md"
    )
    try:
        with open(checklist_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Checklist file not found: {checklist_path}")
        return ""


def get_combined_checklists():
    """Get all checklists combined with separators"""
    return f"""
{load_checklist("opening")}

---

{load_checklist("midgame")}

---

{load_checklist("endgame")}
"""


def get_chess_system_prompt():
    """Get the standard chess system prompt with thinking format instruction"""
    
    return f"""You are a chess engine. Given a chess position, analyze the position and determine the best move.

First, analyze the position inside <think> tags then provide your chosen move in UCI format inside \\boxed{{}} tags


Example format:
<think>
The position appears to be in the [opening/middlegame/endgame] phase. 

1. Safety & Basic Tactics:
- My king is safe on g1, not in check
- No pieces are hanging
- No immediate captures available...

2. [Continue the analysis...]

Based on this analysis, the best move is...
</think>
\\boxed{{f3e5}}"""


def create_chess_user_prompt(board_grid, move_history_str, turn):
    """Create the standard user prompt for chess position analysis"""
    return f"""Current game position:

Current board state:
{board_grid}
Turn: {turn}
Move history (UCI format): {move_history_str}


What is the best move? Analyze the position and provide your answer."""


def format_chess_messages(system_prompt, user_prompt):
    """Format chess prompts as messages for chat template"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]