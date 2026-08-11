"""Pulls real FDA-approved drug labels from the openFDA drug labeling API,
a public, free, no-key-required endpoint, genuinely current government data
with no patient information in it at all.

Saves one raw JSON file per drug to data/raw/, exactly as openFDA returns
it, so chunking.py can be re-run against the same source data without
re-fetching.

Usage:
    ../.venv/bin/python fetch_labels.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

DATA_RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"

# A spread of common prescription drugs across classes, chosen so the
# eval queries in eval_retrieval.py (dosing in renal impairment, drug
# interactions, contraindications, pregnancy, overdose) each have at
# least one real label with genuine content to retrieve.
BRAND_NAMES = [
    "Lipitor",
    "Glucophage",
    "Coumadin",
    "Zestril",
    "Amoxil",
    "Advil",
    "Lopressor",
    "Prilosec",
    "Zoloft",
    "Neurontin",
    "Cozaar",
    "Lasix",
    "Zocor",
    "Norvasc",
    "Ventolin",
]

# A few brand names openFDA's SPL set doesn't carry under that exact
# brand, fall back to a generic-name search for these.
GENERIC_NAME_FALLBACK = {
    "Glucophage": "METFORMIN HYDROCHLORIDE",
    "Coumadin": "WARFARIN SODIUM",
}

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"


def _search(field: str, value: str) -> dict | None:
    params = {"search": f'openfda.{field}:"{value}"', "limit": 1}
    response = requests.get(OPENFDA_LABEL_URL, params=params, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    results = response.json().get("results", [])
    return results[0] if results else None


def fetch_one(brand_name: str) -> dict | None:
    label = _search("brand_name", brand_name)
    if label is not None:
        return label
    fallback_generic = GENERIC_NAME_FALLBACK.get(brand_name)
    if fallback_generic:
        label = _search("generic_name", fallback_generic)
        if label is not None:
            return label
    print(f"  no result for {brand_name}")
    return None


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for brand_name in BRAND_NAMES:
        print(f"fetching {brand_name}...")
        try:
            label = fetch_one(brand_name)
        except requests.RequestException as exc:
            print(f"  request failed: {exc}")
            continue
        if label is None:
            continue
        set_id = label.get("set_id", brand_name.lower())
        out_path = DATA_RAW_DIR / f"{set_id}.json"
        out_path.write_text(json.dumps(label, indent=2))
        fetched += 1
        # openFDA's anonymous rate limit is 240 requests/minute, this is
        # nowhere near that, but a small pause is polite for a public API.
        time.sleep(0.3)

    print(f"\nfetched {fetched} / {len(BRAND_NAMES)} labels into {DATA_RAW_DIR}")


if __name__ == "__main__":
    main()
