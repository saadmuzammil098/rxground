"""Interactive command line chat over the indexed drug labels. Type a
question, get a cited answer or a plain refusal, repeat. Ctrl+C or an
empty line to quit.

Usage:
    ../.venv/bin/python ask.py                  # Ollama, local, free, default
    ../.venv/bin/python ask.py --provider gemini
"""

from __future__ import annotations

import argparse

from rag import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="ollama", choices=["ollama", "gemini", "groq"])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"RxGround baseline RAG, provider={args.provider}. Ask a question about the 15 indexed drugs.")
    print("(Ctrl+C or an empty line to quit.)\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        result = answer_question(question, provider=args.provider, top_k=args.top_k)
        print(f"\n[similarity: {result.best_similarity:.3f}, gate: {'passed' if result.passed_similarity_gate else 'refused, no LLM call'}]")
        print(result.answer)
        print()


if __name__ == "__main__":
    main()
