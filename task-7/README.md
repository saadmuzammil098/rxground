# RxGround Task 7 (bonus): Query Service

Not part of RxGround's original 6-task roadmap, added so
[CareThread](../../carethread), the 30-day roadmap's capstone (Phase 7), can make a
genuine live HTTP call into RxGround for drug-interaction checks instead of a static
lookup table, the actual cross-project integration CareThread Task 3 asks for. Same
shape as FleetPulse's bonus `task-11/`.

## What this is

`service.py` is a thin FastAPI wrapper around `task-3/rag_grounded.py::answer_question`,
already citation-enforced and refusal-capable, unmodified. It adds zero new retrieval or
generation logic: `POST /check_interaction` takes two drug names, builds one question
("Is there a known interaction between X and Y?"), and returns exactly what
`answer_question` would: a refusal if nothing relevant is indexed, or claims with
citations back to the real openFDA label chunk they came from.

## Run it

```bash
pip install -r ../requirements.txt
uvicorn service:app --port 8100
```

Verified live against the real 15-label index:

```
$ curl -X POST localhost:8100/check_interaction \
    -d '{"drug_a": "warfarin", "drug_b": "aspirin"}'
{"refused": true, "refusal_reason": "not covered by the indexed labels", ...}
# neither drug's exact interaction is indexed, refuses rather than
# guessing, exactly the guardrail task-3 built this to prove

$ curl -X POST localhost:8100/check_interaction \
    -d '{"drug_a": "Warfarin Sodium", "drug_b": "ZOCOR"}'
{"refused": false, "claims": [{"text": "...can increase INR levels.",
  "cited_chunk_ids": ["0cbce382-...:drug_interactions:5"], "groundedness": 0.2}], ...}
# both indexed, a real grounded answer citing warfarin's actual label chunk
```

## Why a plain local service, not a Terraform/Lambda deployment

RxGround's own roadmap never assigned it a serving task (Tasks 1-6 are all local
scripts and eval harnesses), and standing up ECR/Lambda/IAM for RxGround here would be
a second, unrelated infrastructure project bolted onto a one-endpoint wrapper. CareThread
Task 3 needs a real HTTP boundary between two separate codebases to call at runtime, not
a hosted, publicly-reachable one, `uvicorn service:app` run alongside CareThread's own
local Floci deployment satisfies that: a genuine live call, same-machine, no cloud
resources needed for the integration itself to be real.

## Why `sys.path` is extended by hand in `service.py`, not via a shared `paths.py`

A helper module named `paths.py` in this folder would collide with `task-3/paths.py`
the instant both get imported in the same process, `import paths` only ever resolves to
whichever one was imported first, silently. This is the identical `src`-package
collision FleetPulse's and CareThread's READMEs already document for the same reason,
just hitting a differently-named module here. `service.py` does its one `sys.path.append`
inline instead.
