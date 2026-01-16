from __future__ import annotations
from typing import Any, Dict
import math

from sgs.sgs_a import sgs_a
from sgs.sgs_i import sgs_i


def compute_refs(instance: Dict[str, Any], seed: int = 0) -> Dict[str, float]:
    """
    Returns reference values used in reward shaping:
      - obj_ref, delay_ref
      - dcap (delay scale cap)
      - eps
    You said "아무값이나 써둬" OK → 여기 기본값은 안전하게 동작하게만 구성.
    """
    _, _, obj_a, delay_a = sgs_a(instance)
    _, _, obj_i, delay_i = sgs_i(instance)

    T = instance["T"]

    if delay_a > 0:
        dcap = min(
            delay_a,
            10 * len(T)
        )
    else:
        dcap = 1

    return {
        "obj_ref": float(obj_i),
        "dcap": float(dcap),
        "eps": 1e-9,
    }
