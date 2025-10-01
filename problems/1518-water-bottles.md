---
id: 1518
slug: water-bottles
title: Water Bottles
source_url: https://leetcode.com/problems/water-bottles/
difficulty: Easy
topics: [math, simulation]
created: 2025-10-01
---

# 1518 · Water Bottles

> **Personal summary (no prompt text copied):**
> Start with `numBottles` full bottles. Each time you drink a bottle you gain one empty bottle.
> You can exchange `numExchange` empty bottles for **one** new full bottle.
> **Goal:** How many bottles of water can you drink in total?

## Approach (Greedy / Math)
Keep a running total of bottles drunk. Repeatedly exchange empties for full bottles while possible:
1. Drink all current full bottles → add to `total`, add to `empties`.
2. Exchange `empties // numExchange` for new full bottles.
3. Update `empties = empties % numExchange + new_full` and loop.

This terminates when `empties < numExchange` and no full bottles remain.

### Python (O(log_{numExchange})(numBottles)) steps, O(1) space)
```python
def water_bottles(numBottles: int, numExchange: int) -> int:
total = 0
full = numBottles
empties = 0
while full > 0:
# drink what you have
total += full
empties += full
full = 0

# exchange empties for new full bottles if possible
trade = empties // numExchange
if trade == 0:
break
empties = empties % numExchange
full = trade

return total
```

### Edge cases
- `numExchange > numBottles`: no exchanges happen → answer is just `numBottles`.
- Large values: operations are simple integer math; no overflow in Python.

### Quick checks
| numBottles | numExchange | expected |
|------------|-------------|----------|
| 9 | 3 | 13 |
| 15 | 4 | 19 |
| 2 | 3 | 2 |

## Notes
- Keep this note as *your* explanation to avoid copying platform text.
- File naming suggestion in repo: `problems/1518-water-bottles.md`.
