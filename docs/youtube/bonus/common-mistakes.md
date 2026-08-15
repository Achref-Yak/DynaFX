# Bonus: 10 Common Mistakes (And How to Fix Them)

**What it does:** Debug the most frequent DynaFX mistakes.

**Duration:** 10-12 minutes

**Hook:** "These are the mistakes I made so you don't have to."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "10 mistakes, 10 fixes."

### 0:30 - Mistakes 1-3: Model Structure (3 min)
1. **Using stock() outside with block** - "Always use the context manager."
2. **Flow expressions without units** - "Always include time units."
3. **Circular references** - "Check your flow dependencies."

### 3:30 - Mistakes 4-6: Knowledge Graph (3 min)
4. **Missing IRI prefixes** - "Always use full IRIs or declare prefixes."
5. **Graph name mismatch** - "Triple must go to the right graph."
6. **SPARQL syntax errors** - "Use PREFIX declarations."

### 6:30 - Mistakes 7-9: Bridge (3 min)
7. **Missing default values** - "params_from_kb needs defaults."
8. **Evidence map type mismatch** - "Score function must return float."
9. **KB_QUERY expression errors** - "Quotes inside quotes break."

### 9:30 - Mistake 10: Integration (1 min)
10. **Running bridge before model is ready** - "Build model first, then connect."

### 10:30 - Quick Reference (1 min)
- "Screenshot this."

---

## Quick Reference Table

| Mistake | Fix |
|---------|-----|
| `s = model.stock(...)` | `with model.stock(...) as s:` |
| `"inflow / time"` | `"inflow units"` |
| Circular flow references | Check dependency order |
| Missing prefix in IRI | Use `<http://...>` |
| Wrong graph name | Check graph parameter |
| SPARQL syntax error | Declare PREFIX first |
| No default in params_from_kb | Add default= parameter |
| Score function returns None | Ensure return float |
| KB_QUERY has nested quotes | Use single quotes outside |
| Bridge before model ready | Build model first |

---

## Thumbnail
- Left: Red X marks
- Right: Green checkmarks
- Bottom: "10 mistakes fixed"

---

## Description
```
Fix the 10 most common DynaFX mistakes.

Docs: https://achref-yak.github.io/DynaFX/
GitHub: https://github.com/Achref-Yak/DynaFX
```

---

## Pin Comment
```
Which mistake have you made? Comment below!
```
