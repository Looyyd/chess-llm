# Chess Move Selection Checklist - Endgame

## 1. Safety & Basic Tactics

- Is my king in check?
- Are any pieces hanging?
- Can I promote a pawn this turn?
- Is my opponent threatening to promote?
- Is there a forced tactical sequence?

## 2. Critical Position Assessment

- **Passed Pawns**
  - Who has passed pawns and how advanced?
  - Can they be stopped or supported?
  - Where can I create a passed pawn?
- **King Position**
  - Is my king active or passive?
  - What are the key squares for both kings?
  - Opposition and zugzwang considerations
- **Pawn Structure**
  - Fixed weaknesses to target
  - Breakthrough possibilities
  - Pawn race calculations

## 3. Strategic Planning (Endgame Focus)

- What's the simplest path to victory (or draw if defending)?
- Should I trade pieces or avoid exchanges?
- Do I need to activate my king or push pawns first?
- What are the critical squares I must control?
- Can I create favorable piece imbalances (good vs bad bishop)?

## 4. Candidate Move Generation

Prioritize moves that:

- Create or advance passed pawns
- Activate the king toward key squares
- Improve piece activity (rooks to 7th rank)
- Force favorable exchanges
- Restrict opponent's pieces
- Set up zugzwang positions

## 5. Calculate & Verify

Pick a final move, out of the candidate moves.
Before making your move:

- Calculate critical lines 5-10+ moves deep
- Verify pawn race counting precisely
- Check resulting positions after exchanges
- Look for stalemate possibilities (if defending)
- Consider waiting moves and triangulation
- Ensure the move advances your plan

If the move doesn't pass these checks redo the checks with another candidate.
