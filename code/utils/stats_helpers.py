#!/usr/bin/env python3
"""
PHTS Theory v3.0 — Statistical helper functions.
"""
import math
import numpy as np
from collections import Counter
from scipy import stats

def shannon_entropy(sequence):
    """Shannon entropy H in bits."""
    counts = Counter(sequence)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

def cross_entropy_h(p_seq, q_seq):
    """H(P, Q) = -sum_x P(x) log Q(x)"""
    p_c = Counter(p_seq)
    q_c = Counter(q_seq)
    total_p = sum(p_c.values())
    total_q = sum(q_c.values())
    eps = 1e-10
    H = 0.0
    for k, cp in p_c.items():
        p = cp / total_p
        q = (q_c.get(k, 0) + eps) / (total_q + eps * len(p_c))
        H -= p * math.log2(q)
    return H

def chi_square_positional(glyph_counts_by_position):
    """
    Chi-square test: are glyph distributions independent of position?
    glyph_counts_by_position: dict of {position: Counter}
    Returns (chi2, p_value)
    """
    from scipy.stats import chi2_contingency
    positions = list(glyph_counts_by_position.keys())
    all_glyphs = sorted(set(
        g for counter in glyph_counts_by_position.values() for g in counter
    ))
    table = []
    for pos in positions:
        row = [glyph_counts_by_position[pos].get(g, 0) for g in all_glyphs]
        table.append(row)
    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p

def cohens_d(group1, group2):
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 + n2 <= 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0

def gzip_compression_ratio(text):
    """
    Estimate compression ratio using gzip.
    Returns percentage of compressed vs original size.
    """
    import gzip
    original = text.encode("utf-8")
    compressed = gzip.compress(original)
    return round(100 * len(compressed) / len(original), 1)
