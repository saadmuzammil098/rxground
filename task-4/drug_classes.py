"""Drug class and brand/generic-name lookup tables for the 15 labels
indexed in task-1.

openFDA's own `pharm_class_epc` field, the obvious source for "drug
class," is present for only 4 of these 15 labels (ZITUVIMET, Advil Dual
Action, Lasix, ZOCOR), too sparse to build a metadata filter on directly,
see task-4's README for the real field-by-field count. This is a small,
curated table instead, standard pharmacology classification, so every one
of the 15 indexed drugs has a class to filter on and a short generic-name
form to expand a brand-name query with.
"""

from __future__ import annotations

# brand_name (as it appears in chunk metadata) -> drug class
DRUG_CLASS: dict[str, str] = {
    "ZITUVIMET": "DPP-4 inhibitor + biguanide combination (diabetes)",
    "Lopressor": "beta blocker",
    "Advil Dual Action with Acetaminophen, Travel BASIX": "NSAID + analgesic combination",
    "Warfarin Sodium": "anticoagulant",
    "Lasix": "loop diuretic",
    "VENTOLIN HFA": "bronchodilator (beta-2 agonist)",
    "Prilosec OTC": "proton pump inhibitor",
    "Norvasc": "calcium channel blocker",
    "Zestril": "ACE inhibitor",
    "ZOCOR": "statin (HMG-CoA reductase inhibitor)",
    "Neurontin": "gabapentinoid / anticonvulsant",
    "COZAAR": "angiotensin receptor blocker (ARB)",
    "Lipitor": "statin (HMG-CoA reductase inhibitor)",
    "Amoxil": "penicillin antibiotic",
    "Zoloft": "SSRI antidepressant",
}

# brand_name -> short generic name(s) used for query expansion. Kept to the
# core active-ingredient word(s) a pharmacist would actually type, not the
# full openFDA generic_name string (e.g. "AMOXICILLIN ORAL SUSP").
GENERIC_NAME: dict[str, list[str]] = {
    "ZITUVIMET": ["sitagliptin", "metformin"],
    "Lopressor": ["metoprolol"],
    "Advil Dual Action with Acetaminophen, Travel BASIX": ["ibuprofen", "acetaminophen"],
    "Warfarin Sodium": ["warfarin"],
    "Lasix": ["furosemide"],
    "VENTOLIN HFA": ["albuterol"],
    "Prilosec OTC": ["omeprazole"],
    "Norvasc": ["amlodipine"],
    "Zestril": ["lisinopril"],
    "ZOCOR": ["simvastatin"],
    "Neurontin": ["gabapentin"],
    "COZAAR": ["losartan"],
    "Lipitor": ["atorvastatin"],
    "Amoxil": ["amoxicillin"],
    "Zoloft": ["sertraline"],
}

# Reverse index, lowercased generic word -> brand name, and lowercased
# brand word -> brand name, both used by query_expansion.py to find which
# names in a query need their counterpart added.
GENERIC_TO_BRAND: dict[str, str] = {
    generic.lower(): brand for brand, generics in GENERIC_NAME.items() for generic in generics
}

# Matched on the brand's first word only ("Advil", not the full "Advil
# Dual Action with Acetaminophen, Travel BASIX"), a pharmacist typing a
# brand name in a query almost never types the whole regulatory label
# name, and every one of these 15 first words is distinct enough on its
# own to identify the drug.
BRAND_LOWER_TO_BRAND: dict[str, str] = {brand.split()[0].lower(): brand for brand in DRUG_CLASS}

ALL_DRUG_CLASSES: list[str] = sorted(set(DRUG_CLASS.values()))
