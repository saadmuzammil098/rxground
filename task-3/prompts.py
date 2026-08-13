"""Prompt templates for task-3's structured, citation-enforced generation.

Two variants of the system prompt exist on purpose, for the deliberate
failure exercise required by the roadmap:

- SYSTEM_PROMPT_ENFORCED demands the strict JSON schema.claims format,
  every claim tagged with a real (set_id, section, chunk_id) citation.
- SYSTEM_PROMPT_UNENFORCED asks the same question over the same retrieved
  context with the citation requirement removed entirely, plain prose, no
  schema, so its output can be compared against the enforced path on the
  same real queries.
"""

from __future__ import annotations

from retrieve_with_ids import RetrievedChunk

NOT_COVERED_PHRASE = "not covered by the indexed labels"

_JSON_SCHEMA_BLOCK = """{
  "refused": <true or false>,
  "refusal_reason": <string, required if refused is true, else null>,
  "claims": [
    {
      "text": "<one factual statement, in your own words>",
      "citations": [
        {"set_id": "<exact set_id from an excerpt below>",
         "section": "<exact section from that excerpt>",
         "chunk_id": "<exact chunk_id from that excerpt>"}
      ]
    }
  ]
}"""

SYSTEM_PROMPT_ENFORCED = (
    "You are RxGround, a clinical drug-reference assistant for a pharmacist. You answer only "
    "using the label excerpts given to you below, never from general knowledge or memory, a "
    "wrong drug interaction or dosing answer here is a safety incident, not a minor mistake.\n\n"
    "Respond with ONLY a single JSON object, no prose before or after it, no markdown code "
    "fences, matching exactly this shape:\n"
    f"{_JSON_SCHEMA_BLOCK}\n\n"
    "Rules:\n"
    "1. Every entry in \"claims\" is one factual statement. Every claim MUST have at least one "
    "citation, and every citation's set_id, section, and chunk_id MUST be copied exactly from "
    "one of the excerpts given below, never invented.\n"
    "2. If the excerpts below do not actually answer the question, for example the drug is not "
    "among them, or the specific detail asked for is not in any excerpt, set \"refused\" to "
    f"true, give a short \"refusal_reason\" ('{NOT_COVERED_PHRASE}' or similarly plain), and "
    "leave \"claims\" empty. Do not guess or extrapolate to avoid refusing.\n"
    "3. Never answer partially confident, if you are not sure a claim is directly supported by "
    "an excerpt below, leave it out rather than including it without a valid citation."
)

SYSTEM_PROMPT_UNENFORCED = (
    "You are RxGround, a clinical drug-reference assistant for a pharmacist. Use the label "
    "excerpts below to answer the question as helpfully as you can, in plain prose. Do not "
    "worry about citing sources or output format, just answer the question directly."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no label excerpts retrieved)"
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[set_id: {chunk.set_id}, chunk_id: {chunk.chunk_id}, section: {chunk.section}, "
            f"drug: {chunk.brand_name} ({chunk.generic_name}), similarity: {chunk.similarity:.3f}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"Label excerpts:\n{format_context(chunks)}\n\nQuestion: {question}"
