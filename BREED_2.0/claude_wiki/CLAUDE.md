# CLAUDE.md — Ionic Conductivity + Solid-State Electrolyte Wiki

## Scope

This wiki is for understanding (1) ML approaches to predicting ionic conductivity in solid-state electrolytes and how well they perform, and (2) the actual bottlenecks in SSE research — what problems people are stuck on.

In scope:
- Inorganic solid electrolytes (oxides, sulfides, halides, garnets, NASICON, argyrodites, perovskites)
- Polymer and composite electrolytes (lower priority but accept)
- ML models for ionic conductivity prediction (GBT, RF, CGCNN, MEGNet, M3GNet, ALIGNN, custom GNNs, etc.)
- First-principles methods used to compute conductivity (AIMD, NEB, BVSE)
- Datasets (OBELiX, Liverpool, Materials Project subsets, custom)

Out of scope:
- Ionic liquids
- Cathode/anode materials beyond cursory mention
- Full cell engineering (manufacturing, scale-up) unless directly relevant to a bottleneck

## Folder structure

```
/raw/                       # source files, untouched
/wiki/
  /sources/                 # one .md per item in raw/
  /concepts/                # one .md per physical/chemical concept
  /methods/                 # one .md per computational or ML method
  open-problems.md          # living doc: bottlenecks and active research questions
  benchmarks.md             # living doc: ML model performance on standard datasets
/outputs/                   # generated artifacts (slides, plots, query results)
CLAUDE.md                   # this file
TODO.md                     # ambiguities flagged during ingestion
```

## Naming conventions

- Filenames are kebab-case, lowercase: `cgcnn.md`, `garnet-electrolytes.md`.
- Cross-references use `[[wikilinks]]`, never markdown links to local files.
- Source notes: kebab-case the first 4-6 words of the paper title. Full title goes in YAML frontmatter.
- If a paper title in `raw/` is truncated and the real title is unclear from the content, add a `_FILENAME_AMBIGUOUS` entry to `TODO.md` and proceed with best guess.

## File schemas

### Source notes (`sources/`)

The schema branches on paper `type`. Reviews are prose-heavy; primary papers are structured. Forcing a survey of 8+ models into the primary-paper template produces bullet dumps that hide the actual signal.

**Frontmatter (all types):**

```yaml
---
title: 
year: 
doi: 
type: [primary | review | dataset | benchmark | other]
material-class: [garnet | sulfide | argyrodite | halide | nasicon | perovskite | polymer | mixed | n/a]
dataset: [obelix | liverpool | hargreaves | materials-project | custom | multi | none | n/a]
target: [sigma | activation-energy | migration-barrier | modulus | diffusivity | band-gap | multi | n/a]
relevance: [high | medium | low]   # to BREED specifically
---
```

`dataset` and `target` are required so the wiki is greppable ("all σ-prediction papers using OBELiX") without reading every note. Use `multi` when a paper spans several. **Do not add fields not listed above** — no `authors`, `venue`, `citations`, etc.

---

**Body — for `type: primary`:**

```markdown
# {Title}

## TL;DR
One paragraph. What did they do, what did they find, why does it matter.

## ML method
- Architecture (e.g. CGCNN, MEGNet, GBT, RF)
- Features / inputs (composition only, structural, both)
- Target variable (room-temp log10 σ, activation energy, etc.)
- Training data: source, size, composition diversity
- Train/test split: random, scaffold, leave-one-cluster-out, time-based
- Performance: metric + value (e.g. MAE = 0.6 log10(S/cm) on held-out test)
- Best prior ML result on same dataset/target (or "none reported")

## Materials / chemistry
- Material class and specific compositions
- Key structural features (Li sublattice, bottleneck size, polyhedra, dopants)

## Bottleneck / problem identified
What does this paper say is hard? If nothing, say so explicitly. If novel, add to `open-problems.md`.

## Relevance to BREED
Concrete. Does this change feature choice? Suggest a new experiment? Validate or contradict the current GBT result? "None" is a valid answer.

## Related
- [[other-source]]
- [[concept]]
- [[method]]
```

---

**Body — for `type: review`:**

```markdown
# {Title}

## TL;DR
One paragraph. What does this review survey and what does it claim?

## Landscape covered
Prose. What methods, materials, properties does it cover? What's the organizing structure of the review?

## Best results cited

| Sub-area | Model | Dataset | Target | Split | Metric | Result | Primary source |
|---|---|---|---|---|---|---|---|

Pull only ML-on-conductivity-adjacent results. Don't transcribe the whole review.

## Bottlenecks identified
Numbered list. The review's claims about what's hard. Each item is a candidate for `open-problems.md`.

## Relevance to BREED
Concrete. Most reviews shift BREED's literature context, not the model itself — say what changed.

## Related
- [[primary-source-cited]]
- [[concept]]
```

For `type: dataset` and `type: benchmark`, write prose freely. These will be rare; don't over-spec until you have one.

### Concept notes (`concepts/`)

```markdown
# {Concept name}

## One-line definition
A single sentence a smart undergrad could understand.

## Why it matters for ionic conductivity
Concrete connection.

## Detailed explanation
2-4 paragraphs. Include math if load-bearing.

## Prerequisites
- [[prereq]]

## Sources discussing this
- [[source-1]]
- [[source-2]]

## Related
- [[adjacent-concept]]
```

### Method notes (`methods/`)

```markdown
# {Method name}

## What it computes / predicts
One sentence.

## How it works
Short technical description.

## Typical performance / cost
Wall time, accuracy ranges, what it gets right and wrong.

## When to use it
Practical guidance.

## Sources using or evaluating this
- [[source-1]]

## Related methods
- [[other-method]]
```

### Living documents

`benchmarks.md` — single table tracking ML model performance. Updated whenever a new ML paper is ingested.

| Source | Model | Dataset | Target | Test setup | Metric | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |

Reviews contribute rows for each cited result that meets the threshold (ML, on a tracked dataset or close analog, with a quantitative metric).

`open-problems.md` — curated list of bottlenecks. Each entry:

```markdown
## {Problem name}
- Description (2-3 sentences)
- Why it's hard
- Approaches tried
- Sources: [[source-1]], [[source-2]]
- Status: [open | partial progress | resolved-ish]
```

When ingesting, dedupe before adding. If a paper raises a bottleneck already in the doc, append the source and merge any new detail into the existing entry. Do not create duplicates.

## Ingestion workflow

When a new file appears in `raw/`:

1. Read the entire source. Do not skim. Do not infer content from the filename alone.
2. Decide the `type`. This determines which body schema to use.
3. Create the source note in `sources/` following the schema for that type. If a frontmatter field is genuinely unknown, leave blank — do not guess.
4. For every load-bearing concept mentioned:
   - If the concept article exists, add this source to its "Sources discussing this" section. Refine the explanation only if this paper adds something new.
   - If it doesn't exist and the concept passes the threshold below, create a stub with the one-line definition and a TODO for fuller treatment.
5. Same logic for methods.
6. **Update `benchmarks.md`**: add a row for every ML quantitative result (one row for primary papers, potentially several for reviews). Use the test setup field aggressively — random splits vs. scaffold splits vs. leave-one-cluster-out are not comparable.
7. **Update `open-problems.md`**: dedupe against existing entries; merge sources into existing problems where applicable.
8. **Fill in `## Related`** in the source note with links to other source notes, concepts, and methods.
9. Flag ambiguities in `TODO.md` with the source filename and the issue. Do not silently guess.

Steps 6, 7, 8 are non-optional. A source note without living-doc updates is a broken ingest.

## Concept threshold

A concept gets its own article when at least one of:
- It is mentioned across 2+ sources.
- It is load-bearing for understanding a single source's core claim.
- It is on the critical path for the BREED next-experiment decision.

If none apply, mention it inline in the source note instead. Do not create a stub.

## Do not

- Do not summarize a paper without reading it. Hallucinated content silently corrupts the wiki.
- Do not merge multiple sources into one note.
- Do not rewrite existing concept articles wholesale. Append and refine; preserve history via git.
- Do not create concept stubs for every noun encountered.
- Do not skip the bottleneck section — "no specific bottleneck identified" is itself useful.
- Do not exceed the schema. If you find yourself wanting a field not listed (authors, venue, citations), leave it out and flag in `TODO.md`.
- Do not equate ML model results across different test splits. Flag the split methodology, always.

## Iteration

This spec is v0. After ingesting the first 5-10 papers:
- Are any schema fields consistently empty? Delete them.
- Are there fields you keep wanting to add manually? Add them to the schema.
- Is the primary/review split working, or are papers landing awkwardly between them?
- Is "Relevance to BREED" producing repetitive entries in the same sub-area? If so, factor it up into a living doc.

Update this file, then re-ingest one paper to verify the new schema works.