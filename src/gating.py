"""
Gating Module — Budget-aware session selection.

Implements two operating modes shown in the architecture overview (Fig. 1):

  Mode a  (explain-all):  Pass-through — every predicted anomaly is explained.
  Mode b  (top-K):        Select the K most uncertain sessions for a given
                          budget ratio B, where K = floor(B * N).

The gating score used in Mode b is the screener uncertainty:

    u(x) = 1 - margin(x)

Higher u(x) means the screener is less certain, so the session is more
likely to benefit from a detailed LLM explanation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple

from .screener import ScreenerOutput
from .data_loader import Session


class GatingMode(str, Enum):
    """Gating operating mode."""
    EXPLAIN_ALL = "explain_all"   # Mode a: pass-through
    TOP_K       = "top_k"         # Mode b: budget-constrained by uncertainty


@dataclass
class GatingConfig:
    """Configuration for the Gating module."""
    mode: GatingMode = GatingMode.EXPLAIN_ALL
    budget: float = 1.0  # fraction of anomalies to explain (0 < B <= 1)

    def __post_init__(self):
        if not 0 < self.budget <= 1.0:
            raise ValueError(f"budget must be in (0, 1], got {self.budget}")
        self.mode = GatingMode(self.mode)


def gate(
    sessions: Sequence[Session],
    screener_outputs: Sequence[ScreenerOutput],
    config: GatingConfig | None = None,
) -> List[Tuple[Session, ScreenerOutput]]:
    """
    Apply gating to a list of (session, screener_output) pairs.

    Only anomaly sessions (screener_output.is_anomaly == True) are
    considered; normal sessions are always dropped before gating.

    Args:
        sessions:         All sessions passed through the screener.
        screener_outputs: Corresponding ScreenerOutput objects.
        config:           GatingConfig (default: explain-all).

    Returns:
        List of (Session, ScreenerOutput) tuples that survived gating,
        sorted by descending uncertainty when mode is TOP_K.
    """
    if config is None:
        config = GatingConfig()  # default: explain-all

    # Filter to anomalies only
    anomaly_pairs: List[Tuple[Session, ScreenerOutput]] = [
        (s, o) for s, o in zip(sessions, screener_outputs) if o.is_anomaly
    ]

    if config.mode == GatingMode.EXPLAIN_ALL or config.budget >= 1.0:
        return anomaly_pairs

    # Mode b: top-K by uncertainty u(x) = 1 - margin
    k = max(1, math.floor(config.budget * len(anomaly_pairs)))
    ranked = sorted(anomaly_pairs, key=lambda pair: pair[1].margin)  # ascending margin = descending uncertainty
    return ranked[:k]
