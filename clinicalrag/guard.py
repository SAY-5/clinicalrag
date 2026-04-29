"""v3: hallucination guard.

A naïve baseline scores how much of the answer's vocabulary appears
in the retrieved evidence. If the answer contains 80% novel tokens
relative to the evidence, the LLM probably fabricated facts that
the retriever didn't surface — flag it.

Real production stacks layer this on top of NLI (does the evidence
*entail* the claim?) and per-claim attribution (split the answer
into atomic claims, score each separately). The shape we expose
here is compatible with that upgrade path: `score_grounding`
returns a single float in [0, 1] and a per-evidence overlap map.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN.finditer(text) if len(m.group(0)) > 2}


@dataclass(frozen=True, slots=True)
class GuardScore:
    score: float                          # overall grounding [0, 1]
    per_evidence_overlap: tuple[float, ...]  # one per evidence chunk


def score_grounding(answer: str, evidence: list[str]) -> GuardScore:
    if not evidence:
        return GuardScore(score=0.0, per_evidence_overlap=())
    answer_toks = _tokenize(answer)
    if not answer_toks:
        return GuardScore(score=0.0, per_evidence_overlap=tuple([0.0] * len(evidence)))
    overlaps: list[float] = []
    grounded_tokens: set[str] = set()
    for ev in evidence:
        ev_toks = _tokenize(ev)
        shared = answer_toks & ev_toks
        overlaps.append(len(shared) / len(answer_toks))
        grounded_tokens |= shared
    overall = len(grounded_tokens) / len(answer_toks)
    return GuardScore(score=overall, per_evidence_overlap=tuple(overlaps))
