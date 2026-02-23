#!/usr/bin/env python3
"""
PHTS Theory v3.0 — Shared corpus loader utilities.
"""
import os
import re
import pandas as pd

SECTION_MAP = {
    "Herbal":         (1, 66),
    "Pharmaceutical": (87, 102),
    "Balneological":  (75, 84),
    "Zodiac":         (70, 73),
    "Astronomical":   (67, 69),
}

def folio_to_section(folio_str):
    m = re.match(r"f(\d+)", folio_str)
    if not m:
        return "Unknown"
    num = int(m.group(1))
    for sec, (start, end) in SECTION_MAP.items():
        if start <= num <= end:
            return sec
    return "Unknown"

def load_processed(data_dir):
    pkl = os.path.join(data_dir, "eva_processed.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError(
            f"Processed corpus not found at {pkl}\n"
            "Run: python code/01_preprocess_eva.py"
        )
    return pd.read_pickle(pkl)
