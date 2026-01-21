# Bug Fixes and Improvements - Summary

## Critical Bugs Fixed

### Bug 1: Incorrect Shape Detection Algorithm (MAJOR)

**Problem:**
The Python implementation was using the regex-based `getShape()` function from the JavaScript code, but the actual JavaScript implementation uses the faster and more accurate `getShapeFast()` function. This was causing significant gameplay issues:
- AI not recognizing free three-in-a-row patterns
- AI not blocking opponent's threats properly
- Overall "dumber" AI behavior

**Root Cause:**
In `/tmp/gobang/src/ai/eval.js`, the code uses `getShapeFast()` which implements a counting-based algorithm, NOT regex patterns. The regex version (`getShape()`) exists in the code but is labeled as slower and less accurate.

**JavaScript Code Structure:**
```javascript
// shape.js has TWO implementations:
getShape()      // Regex-based, slower, simpler to understand
getShapeFast()  // Counting-based, 2x faster, more accurate

// eval.js uses:
const shape = getShapeFast(board, x, y, offsetX, offsetY, role)  // Uses Fast version!
```

**The Fix:**
Completely rewrote `baselines/alpha_beta/shape.py`:
1. Implemented `count_shape()` helper function that counts pieces in one direction
2. Rewrote `get_shape_fast()` to use counting algorithm instead of regex
3. Logic now correctly handles:
   - `no_empty_self_count`: Consecutive pieces with no gaps
   - `one_empty_self_count`: Pieces with at most one gap
   - `side_empty_count`: Empty spaces on the sides
   - Proper detection of open vs blocked patterns

**Impact:**
- AI now correctly recognizes tactical patterns (活三, 活四, 冲四, etc.)
- AI properly blocks opponent threats
- Much stronger gameplay overall

### Bug 2: Hardcoded Bound Value

**Problem:**
The `BoardEvaluator.check_win()` method had a hardcoded check `if count >= 5` instead of using `self.bound`. This meant:
- 6-in-a-row wasn't recognized as a win when bound=10
- AI couldn't play properly with non-standard win conditions

**The Fix:**
1. Added `bound` parameter to `BoardEvaluator.__init__()`
2. Changed `count >= 5` to `count >= self.bound` in check_win()
3. Updated `AlphaBetaWrapper` to pass bound to evaluator

**Note:**
The pattern detection (five, four, three, two) is still optimized for standard 5-in-a-row Gomoku. For bound != 5, win detection works but tactical evaluation may not be optimal.

## Features Added

### Difficulty Levels

Added `--difficulty` option to match the four levels in the React app:

```bash
# Weak (弱智) - depth 2, very fast
python -m tests.human_play --ai_type alpha_beta --difficulty weak

# Easy (简单) - depth 4, fast
python -m tests.human_play --ai_type alpha_beta --difficulty easy

# Medium (中等) - depth 6, balanced
python -m tests.human_play --ai_type alpha_beta --difficulty medium

# Hard (困难) - depth 8, strong but slower
python -m tests.human_play --ai_type alpha_beta --difficulty hard
```

Mapping:
- weak → depth 2 (matches "弱智(2~10层)超快" minimum)
- easy → depth 4 (default)
- medium → depth 6
- hard → depth 8

### Updated Scripts

**tests/human_play.py:**
- Added `alpha_beta` to AI type choices
- Added `--difficulty` option
- Passes depth parameter to wrapper

**tests/evaluator.py:**
- Added `--difficulty` option that overrides `--depth`
- Properly passes depth to both players

## Test Results

All tests pass:

**Shape Detection Tests:**
- ✅ 5-in-a-row detected as FIVE
- ✅ 6-in-a-row detected as FIVE
- ✅ Open three (活三) detected correctly
- ✅ Open four (活四) detected correctly

**Bound Parameter Tests:**
- ✅ Wins detected with bound=5
- ✅ Wins detected with bound=10
- ✅ 6-in-a-row NOT win when bound=10
- ✅ 6-in-a-row IS win when bound=5

## Technical Details

### Shape Detection Algorithm (Counting-Based)

The `getShapeFast()` algorithm works by:

1. **Count in both directions** from the empty position being evaluated
2. **Track multiple counters:**
   - `self_count`: Total pieces of current player
   - `total_length`: Total length of the pattern
   - `no_empty_self_count`: Consecutive pieces without gaps
   - `one_empty_self_count`: Pieces allowing one gap
   - `left_empty`/`right_empty`: Empty spaces on sides

3. **Determine shape based on counts:**
   - FIVE: `no_empty_self_count >= 5`
   - FOUR: `no_empty_self_count == 4` with proper empty spaces
   - THREE: `no_empty_self_count == 3` or `one_empty_self_count == 3` with proper empty spaces
   - TWO: `no_empty_self_count == 2` or `one_empty_self_count == 2`

4. **Distinguish open vs blocked:**
   - Open patterns: Empty spaces on both sides
   - Blocked patterns: Wall/opponent on one or both sides

### Why This Matters

The counting algorithm is more accurate because:
- Handles gaps in patterns correctly (e.g., `XX_X_X` patterns)
- Properly distinguishes between open and blocked patterns
- Faster than regex (2x according to JS comments)
- More nuanced understanding of tactical positions

## Files Modified

- `baselines/alpha_beta/shape.py` - Complete rewrite with counting algorithm
- `baselines/alpha_beta/evaluator.py` - Added bound parameter, fixed check_win
- `wrappers/alpha_beta.py` - Pass bound to evaluator
- `tests/human_play.py` - Added alpha_beta option and difficulty levels
- `tests/evaluator.py` - Added difficulty option

## Verification

The implementation now exactly matches the JavaScript `getShapeFast()` algorithm from lihongxun945/gobang.
