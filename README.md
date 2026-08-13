# RxGround

Clinical drug-reference RAG for a pharmacist looking up FDA-approved drug
labeling data instead of digging through PDFs. Phase 4 of a 30-day
production AI/ML engineering roadmap (see
`30-day-ai-ml-roadmap-industry-portfolio.md` at the repo root), following
GridScribe. Own git repo, own virtual environment, own DVC setup, separate
from FleetPulse's and GridScribe's.

Why this industry pairs with RAG, pharmacy is a clear case for why RAG
needs to be grounded and citation-enforced, a wrong drug interaction answer
is a safety incident, not just a bad user experience. The data is also
genuinely, safely public, FDA drug labeling contains no patient information
at all.

## Architecture

```mermaid
flowchart LR
    subgraph Day14["Day 14, task-1: section-aware chunking vs naive"]
        fda[("openFDA API\n15 real drug labels")] --> raw[("data/raw/*.json")]
        raw --> aware["section-aware chunks"]
        raw --> naive["naive fixed-size chunks"]
        aware --> chromaA[("Chroma:\nsection-aware")]
        naive --> chromaB[("Chroma:\nnaive fixed-size")]
        chromaA --> eval["eval_retrieval.py\n12 real questions"]
        chromaB --> eval
        eval --> results[("top1: 0.667 vs 0.250\ntop3: 0.833 vs 0.500")]
    end

    subgraph Day15["Day 15, task-2: baseline RAG"]
        question[("pharmacist question")] --> gate{"similarity gate\n>= 0.75?"}
        gate -- "no" --> refuse[("not covered,\nno LLM call")]
        gate -- "yes" --> llm["generate()\nOllama / Gemini / Groq"]
        llm --> cited[("cited answer,\n(Brand, section)")]
    end

    subgraph Day16["Day 16, task-3: citation-enforced generation"]
        q3[("pharmacist question")] --> gate2{"similarity gate\n>= 0.75?"}
        gate2 -- "no" --> refuse2[("refused,\nno LLM call")]
        gate2 -- "yes" --> llm2["generate()\nstructured JSON prompt"]
        llm2 --> parse{"valid JSON +\nPydantic schema?"}
        parse -- "no" --> refuse3[("fail closed,\nrefused")]
        parse -- "yes" --> xcheck{"chunk_id in\nretrieved set?"}
        xcheck -- "no" --> invalid[("claim rejected,\ncitation invalid")]
        xcheck -- "yes" --> ground["groundedness.py\nlexical overlap score"]
    end

    chromaA -. "reused unchanged" .-> gate
    chromaA -. "reused unchanged" .-> gate2
```

## Tasks

| Day | Folder | What it is |
|---|---|---|
| 14 | [`task-1/`](./task-1) | Indexes 15 real FDA drug labels (pulled live from the openFDA API) two ways, section-aware chunking versus naive fixed-size chunking, embeds both with BAAI/bge-base-en-v1.5 into separate Chroma collections, and measures the real retrieval accuracy gap on 12 pharmacist-style questions. Section-aware wins 0.667 versus 0.250 top-1 accuracy on the same questions |
| 15 | [`task-2/`](./task-2) | Baseline RAG on top of task-1's index: retrieve, augment, generate with a citation-enforcing prompt and a measured 0.75 similarity gate that refuses questions not covered by the indexed labels before any LLM call. 10/10 real questions were gated correctly, 6/8 in-index answers carried a traceable citation (4/8 in the exact requested format, 2 more in the label's own cross-reference style), and 2/2 out-of-index questions were correctly refused, run live against Ollama and confirmed provider-agnostic against Gemini |
| 16 | [`task-3/`](./task-3) | Citation-enforced generation: structured JSON output, one claim per statement, each claim's `(set_id, section, chunk_id)` citation validated first by Pydantic then cross-checked against the chunk_ids actually retrieved for that question, a claim whose citation does not survive both checks is rejected rather than trusted. A lexical-overlap groundedness score is logged per claim. Run live against Ollama on 18 questions across 3 categories: 10/10 refusal correctness, 0.556 mean citation validity on answered questions, 0.749 mean groundedness, 0/18 parse failures. The real citation-invalid cases were content-correct answers with a fabricated chunk_id (see task-3's README), which is exactly the failure mode the chunk_id cross-check exists to catch. The deliberate failure exercise shows the citation-enforced pipeline correctly refusing an aspirin dosing question (aspirin is not indexed) while the same retrieved context, unenforced, produces a confident-sounding answer built from an unrelated drug's label |

## Tech stack

Click a tool to see what it is used for, and why that one, when there is a
specific reason beyond "it is the roadmap's named default."

<details>
<summary><strong>openFDA drug labeling API</strong>, task-1, source of every drug label indexed</summary>

Public, free, no API key required. Chosen over scraping PDFs because it is
structured, government-published, and genuinely current, and because FDA
labels carry no patient information, so there is no privacy handling to
build for this dataset.
</details>

<details>
<summary><strong>Pydantic</strong>, task-1, the <code>LabelChunk</code> schema and its <code>LabelSection</code> enum</summary>

Same library used across FleetPulse and GridScribe for this repo family.
Rejects a chunk with empty text or an invalid section name before it ever
reaches the embedding step, rather than silently indexing garbage.
</details>

<details>
<summary><strong>sentence-transformers, BAAI/bge-base-en-v1.5</strong>, task-1, turns label text into searchable embeddings</summary>

The specific model the roadmap names for this task. Free, open, and runs
locally, no per-call cost or API key, which matters for indexing hundreds
of label sections.
</details>

<details>
<summary><strong>Chroma</strong>, task-1, stores and searches the two embedding collections</summary>

Picked over the roadmap's other listed option, Qdrant, because it needs no
server or account for a dataset this size, a local folder is enough. Keeps
the day's work focused on the chunking comparison itself, not on running
infrastructure.
</details>

<details>
<summary><strong>DVC + a local Floci S3 bucket</strong>, task-1, versions the 15 raw label JSON files</summary>

Same pattern as FleetPulse and GridScribe: DVC keeps large or frequently
regenerated data out of git history, and Floci is a local AWS emulator, so
this costs nothing and needs no real AWS account.
</details>

<details>
<summary><strong>pytest</strong>, task-1, schema and chunking tests</summary>

Tests run against small inline fixture data, not the real DVC-tracked
labels, on purpose, so they stay fast and work in CI without needing
`dvc pull` or network access.
</details>

<details>
<summary><strong>mermaid-cli, via GitHub Actions</strong>, every README's diagram</summary>

Renders each README's mermaid block through the real parser on every push,
so a diagram that looks fine as plain text but breaks GitHub's renderer
fails the build instead of only showing up broken after merging.
</details>

<details>
<summary><strong>Ollama, qwen2.5:7b</strong>, task-2, the default local generation provider for baseline RAG</summary>

Free, local, no rate limit, which is why every real eval run in task-2's
README uses it. Also the provider on which the citation-format finding
(the model echoes the label's own cross-reference style about as often as
it follows the prompt's requested format) was actually observed.
</details>

<details>
<summary><strong>Gemini, gemini-flash-latest</strong>, task-2, confirmed live as the alternate generation provider</summary>

Run against the same real questions as Ollama specifically to prove the
retrieve-augment-generate pipeline is genuinely provider-agnostic, not
just structured to look that way.
</details>

<details>
<summary><strong>Groq, llama-3.3-70b-versatile</strong>, task-2, one leg of the provider-agnostic generation step</summary>

Wired identically to the other two providers, but not exercised live, no
<code>GROQ_API_KEY</code> is set in this environment, the same documented gap as
GridScribe.
</details>

<details>
<summary><strong>Pydantic</strong>, task-3, <code>Citation</code>/<code>Claim</code>/<code>GroundedResponse</code> structural validation of the LLM's JSON output</summary>

Same library used for task-1's <code>LabelChunk</code> schema, reused here
for a different job, rejecting a claim with no citation or a response
shape that does not match what was asked for, before any citation is
trusted. Necessary but not sufficient on its own, see the chunk_id
cross-check below.
</details>

<details>
<summary><strong>A repo-level chunk_id cross-check</strong>, task-3, catches what Pydantic alone cannot</summary>

Pydantic can confirm a citation has the right shape, it cannot confirm the
cited chunk_id was one that was actually retrieved for the question. This
check does, and it is what caught the real observed failure mode, qwen2.5:
7b giving a factually correct answer while citing a plausible-looking but
fabricated chunk_id, see task-3's README for the exact case.
</details>

<details>
<summary><strong>A lexical-overlap heuristic</strong>, task-3, the groundedness check, deliberately not a second LLM call</summary>

Chosen over LLM-as-judge because the failure being checked for, a claim's
text saying something its cited chunk does not contain, is a fact a
second LLM call could just as easily rubber-stamp as the first one
hallucinated. A deterministic word-overlap score cannot be talked into
agreeing with a fabricated claim.
</details>

## Setup

```bash
cd rxground
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cd task-1
../.venv/bin/dvc pull
../.venv/bin/python index.py
../.venv/bin/python eval_retrieval.py
```
