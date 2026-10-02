"""Deterministic edge policies that never reopen structural ineligibility.

The learned threshold is an engineering decision, not a calibrated probability
of real clinical continuity. Source rule is a control, not boundary ground truth.
"""

import hashlib
import json
import math
import random
from numbers import Real
from typing import Sequence


def select_gates(strategy: str, eligible: Sequence[bool], source_kinds: Sequence[str],
                 scores: Sequence[float] | None = None, threshold: float = 0.9,
                 seed: int = 42, chain_id: str = "") -> list[bool]:
    """Return one boolean per edge for all/source_rule/random/learned.

    ``random`` matches the learned open-edge count independently per source kind
    within this chain, and shuffles only eligible locations. Source strings are
    exact labels: ``source_rule`` opens only ``clinical``. Other nonempty kinds
    remain closed. Randomness uses SHA256 of (seed, chain_id, source_kind), never
    Python's process-dependent hash. Threshold must be fixed before evaluation.
    """
    if strategy not in {"all", "source_rule", "random", "learned"}:
        raise ValueError("strategy must be all, source_rule, random, or learned")
    if isinstance(eligible, (str, bytes)) or isinstance(source_kinds, (str, bytes)):
        raise ValueError("eligible and source_kinds must be sequences")
    flags, kinds = list(eligible), list(source_kinds)
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("eligible must contain booleans, not numeric masks")
    if len(flags) != len(kinds):
        raise ValueError("source_kinds must have one entry per edge")
    if any(not isinstance(kind, str) or not kind.strip() for kind in kinds):
        raise ValueError("source_kinds must contain nonempty strings")
    if (isinstance(threshold, bool) or not isinstance(threshold, Real)
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be a finite number in [0, 1]")
    if type(seed) is not int or not isinstance(chain_id, str):
        raise ValueError("seed must be an integer and chain_id a string")
    probabilities = None
    if scores is not None:
        if isinstance(scores, (str, bytes)):
            raise ValueError("scores must be a probability sequence")
        probabilities = list(scores)
        if len(probabilities) != len(flags):
            raise ValueError("scores must have one entry per edge")
        if any(isinstance(score, bool) or not isinstance(score, Real)
               or not math.isfinite(score) or not 0 <= score <= 1 for score in probabilities):
            raise ValueError("scores must be finite numbers in [0, 1]")
    if strategy == "all":
        return flags
    if strategy == "source_rule":
        return [flag and kind == "clinical" for flag, kind in zip(flags, kinds)]
    if probabilities is None:
        raise ValueError("learned and random policies require scores")
    learned = [flag and score >= threshold for flag, score in zip(flags, probabilities)]
    if strategy == "learned":
        return learned

    output = [False] * len(flags)
    for kind in sorted(set(kinds)):
        positions = [i for i, (flag, source) in enumerate(zip(flags, kinds)) if flag and source == kind]
        count = sum(learned[i] for i in positions)
        identity = json.dumps([seed, chain_id, kind], ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        generator = random.Random(int.from_bytes(digest, "big"))
        for position in generator.sample(positions, count):
            output[position] = True
    return output
