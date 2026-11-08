"""
reasoning.py  (Stage 1)

Three personas produce a rationale for why the preferred response beats the
other, each using a different reasoning strategy.

    CoTReasoner         one chain of thought
    CoTSCReasoner       self-consistency over k sampled chains, then consensus
    SelfRefineReasoner  draft -> self-critique -> refine

Each rationale surfaces the discriminating dimensions (the axes on which the
two responses diverge), because those dimensions seed the principles that
Stage 2 debates. Output for every datapoint is three Rationale objects.
"""

from __future__ import annotations

from collections import Counter
from typing import List

from common import PreferencePair, Rationale, chat, extract_json

# tuning
COT_SC_SAMPLES = 5
SELF_REFINE_STEPS = 2

_BASE_TASK = (
    "You are given a prompt and two candidate responses, A and B. A human "
    "annotator preferred response {pref}. Recover the DECISIVE reason behind "
    "this preference: the underlying rule such that, if someone applied it to "
    "these two responses without being told the label, they would re-select "
    "the preferred response. Focus on concrete, discriminating qualities (the "
    "dimensions on which the two responses actually differ), not qualities they "
    "share. State the reason as something reusable, not a description of these "
    "exact texts."
)

_JSON_INSTRUCTION = (
    'Reply with a single JSON object: '
    '{"reasoning": "<3-6 sentence argument>", '
    '"dimensions": ["<short quality axis>", "..."], '
    '"questions": [{"select_prompt": "Select the response that ...", '
    '"reason": "<why this criterion drove the preference>"}]}. '
    "Each dimension is a short noun phrase naming a comparable quality "
    '(for example "tension and conflict", "logical coherence", "specificity '
    'of evidence"). Return 2 to 5 dimensions. Each question is a candidate '
    'selection criterion beginning exactly with "Select the response". Return '
    "3 to 7 questions covering your dimensions. No text outside the JSON."
)


def _pair_block(pair: PreferencePair) -> str:
    return (
        f"PROMPT:\n{pair.prompt}\n\n"
        f"RESPONSE A:\n{pair.response_a}\n\n"
        f"RESPONSE B:\n{pair.response_b}\n\n"
        f"PREFERRED: {pair.preferred.upper()}"
    )


def _parse(pair: PreferencePair, strategy: str, raw: str) -> Rationale:
    obj = extract_json(raw)
    reasoning = (obj.get("reasoning") or raw).strip()
    dims = obj.get("dimensions") or []
    dims = [str(d).strip().lower() for d in dims if str(d).strip()]
    qs = []
    for q in obj.get("questions") or []:
        if isinstance(q, dict):
            sp = str(q.get("select_prompt", "")).strip()
        else:
            sp = str(q).strip()
        if sp.lower().startswith("select the response"):
            qs.append({"select_prompt": sp,
                       "reason": str(q.get("reason", "")).strip()
                       if isinstance(q, dict) else ""})
    return Rationale(pair.id, strategy, pair.preferred.upper(), reasoning, dims, qs)


class CoTReasoner:
    strategy = "cot"

    def reason(self, pair: PreferencePair) -> Rationale:
        sys = _BASE_TASK.format(pref=pair.preferred.upper())
        user = (
            _pair_block(pair)
            + "\n\nThink step by step through the differences, then conclude. "
            + _JSON_INSTRUCTION
        )
        raw = chat(
            [{"role": "system", "content": sys},
             {"role": "user", "content": user}],
            temperature=0.4,
        )
        return _parse(pair, self.strategy, raw)


class CoTSCReasoner:
    """Self-consistency: sample k independent chains, keep the dimensions that
    recur across a majority of chains, then synthesize one consensus rationale
    grounded only in those stable dimensions."""

    strategy = "cot_sc"

    def __init__(self, samples: int = COT_SC_SAMPLES):
        self.samples = samples

    def reason(self, pair: PreferencePair) -> Rationale:
        sys = _BASE_TASK.format(pref=pair.preferred.upper())
        user = (
            _pair_block(pair)
            + "\n\nReason step by step, then conclude. " + _JSON_INSTRUCTION
        )
        chains: List[Rationale] = []
        for _ in range(self.samples):
            raw = chat(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": user}],
                temperature=0.9,
            )
            chains.append(_parse(pair, self.strategy, raw))

        # majority vote over dimensions
        counter: Counter = Counter()
        for c in chains:
            counter.update(set(c.dimensions))
        threshold = max(1, self.samples // 2)
        stable = [d for d, n in counter.most_common() if n >= threshold]
        if not stable:
            stable = [d for d, _ in counter.most_common(3)]

        # synthesize a consensus rationale constrained to the stable dimensions
        joined = "\n".join(f"- {c.reasoning}" for c in chains)
        synth_user = (
            f"{self.samples} independent analyses were produced for the same "
            f"preference judgement.\n\nANALYSES:\n{joined}\n\n"
            f"DIMENSIONS THAT RECUR ACROSS A MAJORITY: {stable}\n\n"
            "Write one consensus rationale that relies only on the recurring "
            "dimensions and discards idiosyncratic points. "
            + _JSON_INSTRUCTION
        )
        raw = chat(
            [{"role": "system", "content": sys},
             {"role": "user", "content": synth_user}],
            temperature=0.2,
        )
        consensus = _parse(pair, self.strategy, raw)
        # prefer the majority-voted dimensions if the synthesizer drifted
        if stable:
            consensus.dimensions = stable
        return consensus


class SelfRefineReasoner:
    """Draft, then iteratively self-critique and refine the rationale so that
    every claim is discriminating and specific."""

    strategy = "self_refine"

    def __init__(self, steps: int = SELF_REFINE_STEPS):
        self.steps = steps

    def reason(self, pair: PreferencePair) -> Rationale:
        sys = _BASE_TASK.format(pref=pair.preferred.upper())
        block = _pair_block(pair)

        raw = chat(
            [{"role": "system", "content": sys},
             {"role": "user", "content": block + "\n\nWrite a first-draft "
              "rationale. " + _JSON_INSTRUCTION}],
            temperature=0.6,
        )
        current = _parse(pair, self.strategy, raw)

        for _ in range(self.steps):
            critique = chat(
                [{"role": "system", "content": "You are a strict reviewer of "
                  "preference rationales."},
                 {"role": "user", "content":
                  f"{block}\n\nDRAFT RATIONALE:\n{current.reasoning}\n"
                  f"DRAFT DIMENSIONS: {current.dimensions}\n\n"
                  "List concrete weaknesses: vague claims, points that are not "
                  "discriminating (true of both responses), and missing "
                  "high-tension differences. Be brief."}],
                temperature=0.3,
            )
            raw = chat(
                [{"role": "system", "content": sys},
                 {"role": "user", "content":
                  f"{block}\n\nPREVIOUS RATIONALE:\n{current.reasoning}\n\n"
                  f"REVIEWER CRITIQUE:\n{critique}\n\n"
                  "Revise the rationale to fix every weakness. Keep only "
                  "discriminating, specific points. " + _JSON_INSTRUCTION}],
                temperature=0.4,
            )
            current = _parse(pair, self.strategy, raw)
        return current


def run_reasoning(pair: PreferencePair) -> List[Rationale]:
    """Run all three personas on one datapoint."""
    return [
        CoTReasoner().reason(pair),
        CoTSCReasoner().reason(pair),
        SelfRefineReasoner().reason(pair),
    ]


if __name__ == "__main__":
    demo = PreferencePair(
        id="demo-1",
        prompt="Write the opening of a story about two siblings dividing an inheritance.",
        response_a=("The lawyer slid the folder across the table. Mara did not "
                    "reach for it. 'You already spent it,' she said, and her "
                    "brother's jaw tightened before he could arrange his face."),
        response_b=("It was a sunny day. The two siblings were happy to receive "
                    "their inheritance and they decided to share everything "
                    "fairly and equally, because family matters most."),
        preferred="A",
    )
    for r in run_reasoning(demo):
        print(f"[{r.strategy}] dims={r.dimensions}\n{r.reasoning}\n")
