"""
run_pipeline.py

End-to-end driver:
    reasoning (Stage 1) -> debate (Stage 2) -> clustering (Stage 3)

Reads preference pairs from a JSON file, runs the three stages, and writes
per-datapoint rationales and principles plus the final non-overlapping
constitution to ./out.

Usage
    python run_pipeline.py --data data_sample.json --out out
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from typing import List, Optional

from common import PreferencePair, Principle
from reasoning import run_reasoning
from debate import run_debate
from clustering import (
    cluster_principles,
    validate_constitution,
    minimize_constitution,
)


CSV_REQUIRED = ("chosen", "rejected", "task_prompt")
CSV_META = ("dataset", "task", "score_label", "score_chosen", "score_rejected",
            "novelty_chosen", "surprise_chosen", "diversity_chosen", "quality_chosen")


def _load_pairs_csv(path: str, limit: int = 0, source_dataset: Optional[str] = None,
                    seed: int = 0, sample: bool = False) -> List[PreferencePair]:
    """Read preference pairs from a CSV with columns
        dataset, task, score_label, score_chosen, score_rejected,
        chosen, rejected, ..., task_prompt
    `chosen` is the human-preferred response. Which side (A or B) holds the
    chosen text is randomised per row with a fixed seed, so the blind
    reconstruction judge cannot exploit position, and runs stay reproducible."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    missing = [c for c in CSV_REQUIRED if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}. "
                         f"Found: {list(rows[0])}")

    if source_dataset:
        rows = [r for r in rows if (r.get("dataset") or "").strip() == source_dataset]
        if not rows:
            raise ValueError(f"no rows with dataset == '{source_dataset}'")

    # drop duplicate (prompt, chosen, rejected) triples, then assign stable ids
    seen, uniq = set(), []
    for r in rows:
        key = (r["task_prompt"], r["chosen"], r["rejected"])
        if key not in seen and r["chosen"].strip() and r["rejected"].strip():
            seen.add(key)
            r["_pid"] = f"{(r.get('dataset') or 'rq').strip()}-{len(uniq):04d}"
            uniq.append(r)

    if sample and limit and limit < len(uniq):
        uniq = random.Random(seed).sample(uniq, limit)
    elif limit:
        uniq = uniq[:limit]

    pairs = []
    for r in uniq:
        pid = r["_pid"]
        swap = random.Random(f"{seed}:{pid}").random() < 0.5
        pairs.append(PreferencePair(
            id=pid,
            prompt=r["task_prompt"].strip(),
            response_a=(r["rejected"] if swap else r["chosen"]).strip(),
            response_b=(r["chosen"] if swap else r["rejected"]).strip(),
            preferred=("B" if swap else "A"),
            meta={k: r[k] for k in CSV_META if k in r},
        ))
    return pairs


def load_pairs(path: str, limit: int = 0, source_dataset: Optional[str] = None,
               seed: int = 0, sample: bool = False) -> List[PreferencePair]:
    """CSV (chosen/rejected schema) or JSON (explicit A/B schema)."""
    if path.lower().endswith(".csv"):
        return _load_pairs_csv(path, limit, source_dataset, seed, sample)
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if limit:
        rows = rows[:limit]
    return [PreferencePair(**row) for row in rows]


_OCCUPATION = re.compile(
    r"Occupation Category:\s*(.*?)\s+(?:Detailed Job Description|Income)\s*:", re.S)


def _agent_name(cell: str, idx: int) -> str:
    """Derive a debate agent name from a persona cell.
    A full attribute profile ("Age: 42 Sex: ... Occupation Category: X ...")
    is named from its Occupation Category. Otherwise "Name | identity" or
    "Name: identity" gives an explicit name, then the text before " who ",
    then Persona_<idx>."""
    cell = cell.strip()
    m = _OCCUPATION.search(cell)
    if m:
        head = m.group(1).split(",")[0]
    else:
        for sep in ("|", ":"):
            if sep in cell:
                head = cell.split(sep, 1)[0]
                break
        else:
            head = cell.split(" who ", 1)[0] if " who " in cell else ""
    head = re.sub(r"^(a|an|the)\s+", "", head.strip(), flags=re.I)
    words = re.findall(r"[A-Za-z0-9]+", head)[:3]
    return "_".join(w.capitalize() for w in words) if words else f"Persona_{idx}"


def _identity(cell: str) -> str:
    """Full profiles are passed through whole; 'Name | identity' is split."""
    cell = cell.strip()
    if _OCCUPATION.search(cell):
        return cell
    for sep in ("|", ":"):
        if sep in cell:
            return cell.split(sep, 1)[1].strip()
    return cell


def load_personas(csv_path: str, dataset: Optional[str] = None):
    """Read personas from a CSV whose first column is the dataset name and
    whose remaining columns are persona_1, persona_2, persona_3.

        dataset,persona_1,persona_2,persona_3
        creative_writing,A literature professor who ...,A screenwriter who ...,A literary critic who ...

    persona_1 -> cot, persona_2 -> cot_sc, persona_3 -> self_refine.
    If dataset is omitted, the first data row is used."""
    if not csv_path:
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no data rows")
    if any(None in r for r in rows):
        raise ValueError(
            f"{csv_path}: a row has more fields than the header. Persona cells "
            "containing commas must be wrapped in double quotes.")

    key = list(rows[0].keys())[0]          # first column, whatever it is called
    if dataset:
        wanted = dataset.strip().lower()
        matches = [r for r in rows if (r[key] or "").strip().lower() == wanted]
        if not matches:
            names = ", ".join(sorted((r[key] or "").strip() for r in rows))
            raise ValueError(f"dataset '{dataset}' not in {csv_path}. Available: {names}")
        row = matches[0]
    else:
        row = rows[0]

    strategies = ["cot", "cot_sc", "self_refine"]
    personas = {}
    for i, strat in enumerate(strategies, start=1):
        cell = (row.get(f"persona_{i}") or "").strip()
        if not cell:
            raise ValueError(f"row '{row[key]}' is missing persona_{i}")
        personas[strat] = (_agent_name(cell, i), _identity(cell))
    return personas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_sample.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-cycles", type=int, default=4)
    ap.add_argument("--min-principles", type=int, default=3,
                    help="judge may not finalize a list shorter than this")
    ap.add_argument("--votes", type=int, default=1,
                    help="reconstruction trials per check (majority vote)")
    ap.add_argument("--k", type=int, default=None,
                    help="fix the number of clusters (default: silhouette)")
    ap.add_argument("--k-max", type=int, default=12,
                    help="upper bound of the silhouette sweep")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N pairs (0 = all)")
    ap.add_argument("--sample", action="store_true",
                    help="randomly sample --limit pairs instead of taking the first N")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for A/B side assignment and sampling")
    ap.add_argument("--source-dataset", default=None,
                    help="filter the data CSV by its 'dataset' column")
    ap.add_argument("--personas", default=None,
                    help="CSV with first column = dataset name and columns "
                         "persona_1, persona_2, persona_3")
    ap.add_argument("--dataset", default=None,
                    help="dataset name to select from the personas CSV "
                         "(default: first row)")
    ap.add_argument("--minimize-sample", type=int, default=0,
                    help="evaluate --minimize on N random pairs instead of all "
                         "(cost is principles x pairs x votes)")
    ap.add_argument("--minimize", action="store_true",
                    help="greedily trim the constitution to the smallest set "
                         "that still reconstructs preferences")
    ap.add_argument("--allow-partial", action="store_true",
                    help="cluster even if some datapoints are missing principles")
    ap.add_argument("--resume", action="store_true",
                    help="skip datapoints already in out/per_datapoint.jsonl")
    ap.add_argument("--stages", choices=["all", "reason-debate", "cluster"],
                    default="all",
                    help="reason-debate: stop after the debate (checkpointed). "
                         "cluster: skip to clustering using the checkpoint.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tdir = os.path.join(args.out, "transcripts")
    os.makedirs(tdir, exist_ok=True)
    ckpt = os.path.join(args.out, "per_datapoint.jsonl")

    verbose = not args.quiet
    pairs = load_pairs(args.data, limit=args.limit,
                       source_dataset=args.source_dataset,
                       seed=args.seed, sample=args.sample)
    n_a = sum(p.preferred == "A" for p in pairs)
    print(f"[data] {len(pairs)} pairs | preferred=A {n_a}, B {len(pairs)-n_a}")
    est = len(pairs) * (12 + args.max_cycles * (4 + 2 * args.votes))
    print(f"[data] rough LLM call budget for stages 1-2: ~{est}")

    personas = load_personas(args.personas, args.dataset)
    if personas and verbose:
        print("[personas]")
        for strat, (name, ident) in personas.items():
            print(f"  {strat:12} -> {name}: {ident[:70]}")

    # resume: replay completed datapoints from the checkpoint file
    per_point = []
    done = set()
    if os.path.exists(ckpt):
        with open(ckpt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                per_point.append(rec)
                done.add(rec["id"])
        if args.stages == "cluster":
            print(f"[cluster-only] loaded {len(done)} datapoints from {ckpt}")
        elif args.resume:
            print(f"[resume] {len(done)} datapoints already done, skipping them")
        else:
            print(f"[warn] {ckpt} exists with {len(done)} records. "
                  f"Pass --resume to continue, or use a fresh --out directory.")
            sys.exit(1)
    elif args.stages == "cluster":
        print(f"[error] --stages cluster needs an existing {ckpt}")
        sys.exit(1)

    if args.stages != "cluster":
        ck = open(ckpt, "a", encoding="utf-8")
        for pair in pairs:
            if pair.id in done:
                continue
            print(f"\n=== datapoint {pair.id} ===")
            rationales = run_reasoning(pair)                    # Stage 1
            principles = run_debate(pair, rationales,           # Stage 2
                                    max_cycles=args.max_cycles,
                                    votes=args.votes,
                                    personas=personas,
                                    transcript_path=os.path.join(tdir, f"{pair.id}.txt"),
                                    min_principles=args.min_principles,
                                    verbose=verbose)
            rec = {
                "id": pair.id,
                "prompt": pair.prompt,
                "preferred": pair.preferred,
                "preferred_text": pair.preferred_text(),
                "rejected_text": pair.other_text(),
                "meta": pair.meta,
                "rationales": [r.as_dict() for r in rationales],
                "principles": [p.as_dict() for p in principles],
            }
            per_point.append(rec)
            # checkpoint immediately: a crash after this loses nothing
            ck.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ck.flush()
            os.fsync(ck.fileno())
        ck.close()

    all_principles: List[Principle] = [
        Principle(**p) for rec in per_point for p in rec["principles"]
    ]
    if args.stages == "reason-debate":
        print(f"\nStages 1-2 complete. {len(all_principles)} principles across "
              f"{len(per_point)} datapoints checkpointed to {ckpt}")
        print(f"Transcripts in {tdir}/. Rerun with --stages cluster to cluster.")
        return

    # Stage 3 pools principles from EVERY datapoint, so make sure every
    # requested pair actually produced one before clustering.
    have = {rec["id"] for rec in per_point}
    covered = [p for p in pairs if p.id in have]
    missing = [p.id for p in pairs if p.id not in have]
    if missing and not args.allow_partial:
        print(f"\n[error] {len(missing)} of {len(pairs)} requested datapoints have "
              f"no principles (e.g. {missing[:3]}). Clustering must run over all "
              f"datapoints. Rerun with --resume to finish them, or pass "
              f"--allow-partial to cluster anyway.")
        sys.exit(1)
    if missing:
        print(f"\n[warn] clustering over {len(covered)}/{len(pairs)} datapoints "
              f"(--allow-partial)")
    extra = len(have) - len(covered)
    if extra:
        print(f"[warn] checkpoint holds {extra} datapoints outside the requested "
              f"set; they are pooled into clustering too")

    print(f"\n[stage 3] pooling {len(all_principles)} principles from "
          f"{len(per_point)} datapoints")
    constitution = cluster_principles(all_principles, k=args.k,
                                      k_max=args.k_max, verbose=verbose)  # Stage 3
    principle_texts = [c["principle"] for c in constitution]

    # reconstruction: does the constitution recover the original preferences?
    eval_pairs = covered or pairs
    report = validate_constitution(eval_pairs, principle_texts, votes=args.votes)
    print(f"\n[reconstruction] constitution accuracy: "
          f"{report['accuracy']:.2f} over {report['n']} datapoints")

    if args.minimize:
        mpairs = eval_pairs
        if args.minimize_sample and args.minimize_sample < len(eval_pairs):
            mpairs = random.Random(args.seed).sample(eval_pairs, args.minimize_sample)
        cost = len(principle_texts) * len(mpairs) * args.votes
        print(f"[minimize] ~{cost} calls over {len(mpairs)} pairs "
              f"({len(principle_texts)} principles x {args.votes} votes)")
        principle_texts = minimize_constitution(
            mpairs, principle_texts, votes=args.votes, verbose=verbose)
        report = validate_constitution(eval_pairs, principle_texts, votes=args.votes)
        constitution = [c for c in constitution if c["principle"] in principle_texts]
        print(f"[reconstruction] minimised to {len(principle_texts)} principles, "
              f"accuracy {report['accuracy']:.2f}")

    with open(os.path.join(args.out, "per_datapoint.json"), "w", encoding="utf-8") as f:
        json.dump(per_point, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "constitution.json"), "w", encoding="utf-8") as f:
        json.dump({"principles": constitution, "reconstruction": report},
                  f, indent=2, ensure_ascii=False)

    print("\n=== FINAL CONSTITUTION ===")
    for i, c in enumerate(constitution, 1):
        print(f"{i}. {c['principle']}  (support={c['support']})")
    print(f"\nWrote outputs to {args.out}/")


if __name__ == "__main__":
    main()