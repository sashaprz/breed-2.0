---
name: Ismat
date: 2026-05-24
type: expert-interview
topics: [ceramic-cost, ceramic-manufacturing, polymer-synthesis, computational-tools-accessibility]
---

# Expert Interview — Ismat (May 24, 2026)

Informal conversation. Notes recorded by Sasha Przybylski. Not a published source — do not cite in benchmarks.md or treat as peer-reviewed data.

---

## LLZO cost and ceramic scalability

LLZO pellets (14 mm diameter, 0.7 mm thick) cost **$181.95 per disk** commercially. This is a direct barrier to scaling — not a theoretical concern. Ceramics in general are expensive to manufacture because they require **extremely high temperatures maintained at very precise tolerances**. The energy and equipment demands make large-scale production economically unviable at current prices.

## Solid-solid interface problem (ceramics)

Ceramics have a fundamentally worse interface problem than polymers because the contact is solid-solid. Polymers can conform to electrode surfaces; ceramics cannot. Ismat flagged this as a key practical limitation, consistent with the literature on interface resistance, but grounded in hands-on experience.

See also: [[open-problems#bulk-conductivity-device-performance-interface-resistance-gap]]

## Polymer SSE synthesis: the tuning problem

For polymers, the core difficulty is **synthesis control** — figuring out the synthesis methods and conditions that reliably produce the properties and performance you actually want. This is a process engineering challenge as much as a materials science one. It's not that high-performance polymer SSEs don't exist in principle; it's that translating a known composition into a reliable, reproducible process is hard.

## Computational tools: accessibility and adoption

Ismat said he would have used computational tools (e.g., BVSE, AIMD, ML screening) in his work, but two barriers prevented it:
1. **Compute budget** — he didn't have access to sufficient computational resources.
2. **Learning curve** — didn't want to invest the time to learn the tools himself.

His direct quote (paraphrased): *"If it was easier to use, 1000% [I would have used it]."*

This is a first-person data point on the adoption gap between tool capability and experimental researcher uptake. The tools exist; the bottleneck is usability and accessibility, not fundamental scientific gaps.
