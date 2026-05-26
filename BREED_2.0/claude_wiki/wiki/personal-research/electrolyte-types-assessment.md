---
author: Sasha Przybylski
date: 2026-05-25
type: personal-research
status: surface-level opinion — not peer-reviewed, not published
topics: [ceramics, sulfides, polymers, manufacturing, market-fit]
---

# Electrolyte Types — Personal Assessment

Not a published source. My own surface-level thinking on where each class lands practically.

---

## Ceramics

Seem pretty much off the charts for most applications — the manufacturing cost is prohibitive (see [[interviews/ismat-may2026]] for the LLZO $181.95/disk data point). The niche where ceramics make sense is anywhere **safety is the overriding constraint** (aerospace, medical implants, military) where cost is secondary. Not a realistic candidate for mass-market applications unless cost collapses dramatically.

## Sulfides

Actually look promising. Key properties:
- Require **lower sintering temperatures** than ceramics → less energy-intensive manufacturing, lower cost ceiling
- Still harder to manufacture than polymers, but the gap to ceramics is large
- Main downside: **water sensitivity** — sulfide SSEs degrade on contact with moisture, requiring dry-room or inert-atmosphere processing throughout manufacturing and assembly

Most of the well-performing sulfides are **crystalline** — the high-conductivity candidates (LGPS, argyrodites) are ordered crystal structures, not amorphous glasses.

**Tentative view:** sulfides seem like the most likely near-term replacement for liquid electrolytes in **EVs**, where the investment in dry-room infrastructure is justifiable at scale and safety/energy-density premium is valued.

## Polymers

Easiest to manufacture — processable at low temperatures, compatible with roll-to-roll and existing film manufacturing infrastructure. Main challenge is **tuning**: getting the synthesis conditions right to hit target conductivity, mechanical properties, and stability simultaneously (see [[personal-research/synthesis-pathway-prediction]]).

**Tentative view:** polymers are the likely eventual winner for **consumer electronics** — thin, flexible, low-cost, safe. But the synthesis-property control problem needs to be solved first.

---

## Summary table (personal view)

| Class | Manufacturing difficulty | Cost | Main barrier | Most likely application |
|---|---|---|---|---|
| Ceramics | Very high (high T, precise) | Very high | Cost | Safety-critical niches only |
| Sulfides | Medium (lower T than ceramics) | Medium | Water sensitivity | EVs |
| Polymers | Low | Low | Synthesis tuning | Consumer electronics |
