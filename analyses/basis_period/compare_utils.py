"""Helpers shared by the basis-period comparison: reach attributes, top decile and allocation comparison.

Ensemble allocations are equal-member reachwise medians after each member is matched to 10 Gm3/yr, as in the paper.
Priorities follow the manuscript (Figure 5d): top decile = floor(0.1 N) reaches by effort. Reference mean flow and
drainage area are read from the reach attributes of the outputs_common archive (read only).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import data  # noqa: E402

CORE = data("common") / "reach_attributes.csv"


def top(values, k):
    return set(np.argsort(-values, kind="stable")[:k])


def compare(a, b, k):
    ta, tb = top(a, k), top(b, k)
    return dict(spearman=float(spearmanr(a, b).statistic), top_decile_overlap=100 * len(ta & tb) / k)
