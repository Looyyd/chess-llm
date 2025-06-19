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
    all_checklists = get_combined_checklists()
    
    return f"""You are a chess engine. Given a chess position, analyze the position and determine the best move.

First, analyze the position inside <think> tags, using the following checklists to guide your thinking:

{all_checklists}

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


def create_chess_user_prompt(board_grid, move_history_str, turn):
    """Create the standard user prompt for chess position analysis"""
    return f"""Current game position:

Move history (UCI format): {move_history_str}
Turn: {turn}

Current board state:
{board_grid}

What is the best move? Analyze the position and provide your answer."""


def format_chess_messages(system_prompt, user_prompt):
    """Format chess prompts as messages for chat template"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]