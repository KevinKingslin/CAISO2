"""Original DICAI debate with reconstruction feedback and finalization gates."""

from caiso.core.recovery import ModelResponseError, RetryExhausted, response_failure
from caiso.prompts import render_messages

from .personas import Persona
from .preferences import candidate_views, task_view
from .reconstruction import reconstructs
from .parsing import dedupe, explicitly_unchanged, parse_principles
from .parsing import principle_key as principle_key


def render(values):
    return "\n".join(f"- Q{i}: {value}" for i, value in enumerate(values, 1))


def debate(pair, catalog, providers, role, seed, rounds=4, config=None, progress=None):
    config = config or {}
    history_turns = config.get("history_turns", 8)
    temperatures = config.get("temperatures", {})
    minimum = config.get("min_principles", 3)
    minimum_rounds = config.get("min_debate_rounds", 1)
    votes = config.get("reconstruction_votes", 1)
    if rounds < 1 or len(catalog) != 3:
        raise ValueError("DICAI requires three debaters and at least one round")
    working = dedupe([p for entry in catalog for p in entry.get("principles", [])])
    history = [{"speaker": "initial_working_list", "content": render(working)}]
    a, b = candidate_views(pair)
    evidence = {
        "context": task_view(pair.context), "response_a": a, "response_b": b,
        "preferred": pair.preferred, "preferred_text": a if pair.preferred == "A" else b,
        "min_principles": minimum, "total_cycles": rounds,
    }

    def recent(turns=None):
        entries = [h for h in history if h["speaker"] not in {"initial_working_list", "reconstruction", "chairperson_format_error"}]
        return "\n\n".join(
            f"[{turn['speaker']}]\n{turn['content']}" for turn in entries[-(history_turns if turns is None else turns):]
        ) or "(no debate history yet)"

    def check(cycle, when):
        if progress:
            progress(cycle, "reconstruction " + when)
        result = reconstructs(providers, role, pair, working, votes, temperatures.get("reconstruction", 0.0))
        history.append({"speaker": "reconstruction", "cycle": cycle, "when": when, "result": result})
        if not working:
            return result, "RECONSTRUCTION STATUS: the working list is empty."
        return result, (
            "RECONSTRUCTION STATUS: applying ONLY these questions selects Trajectory "
            f"{result['predicted']}; the provided preference label favors Trajectory {pair.preferred}; "
            f"reconstructed = {'YES' if result['reconstructed'] else 'NO'}."
        )

    finalized = published = False
    rejection_kind = ""
    result = {"reconstructed": False}
    for cycle in range(1, rounds + 1):
        result, status = check(cycle, "before")
        fields = {
            **evidence, "working_list": render(working) or "(the working list is empty)",
            "reconstruction_status": status, "cycle": cycle, "question_count": len(working),
        }
        for i, entry in enumerate(catalog):
            if progress:
                progress(cycle, f"debater {i + 1}/3")
            persona = Persona.model_validate(entry["persona"])
            text = providers.complete(
                role, render_messages(
                    "dicai", "debate.debater" + rejection_kind, **fields,
                    agent_name=persona.identity, identity=persona.prompt(), trace=entry["reasoning"],
                    seeds="\n".join(f"- {p}" for p in entry.get("principles", [])) or "(none)",
                    debate_history=recent(),
                ),
                seed=seed + (cycle - 1) * 4 + i,
                temperature=temperatures.get("debater", 0.7),
                response_context={"pair_id": pair.pair_id, "stage": "debater", "cycle": cycle, "debater": i + 1},
            )
            history.append({"speaker": f"debater_{i + 1}", "content": text, "cycle": cycle})
        if progress:
            progress(cycle, "chairperson")

        def valid_update(text):
            if working and explicitly_unchanged(text):
                return text
            parse_principles(text)
            return text

        try:
            chair = providers.complete(
                role, render_messages(
                    "dicai", "debate.chairperson" + rejection_kind, **fields, cycle_arguments=recent(3),
                ),
                seed=seed + (cycle - 1) * 4 + 3, validate=valid_update,
                temperature=temperatures.get("chairperson", 0.2),
                response_context={"pair_id": pair.pair_id, "stage": "chairperson", "cycle": cycle},
            )
        except (ModelResponseError, RetryExhausted) as error:
            if not response_failure(error):
                raise
            history.append({
                "speaker": "chairperson_format_error", "cycle": cycle,
                "content": "The chairperson returned no usable updated list. "
                + ("The last validated working list was retained." if working else "The working list remains empty."),
                "error": str(error), "retained_principles": list(working),
            })
            finalized = False
            continue
        history.append({"speaker": "chairperson", "content": chair, "cycle": cycle})
        published = True
        if not explicitly_unchanged(chair):
            working = parse_principles(chair)
        result, _ = check(cycle, "after")
        declared_over = "DEBATE_OVER" in chair.upper()
        finalized = declared_over and result["reconstructed"] and len(working) >= minimum and cycle >= minimum_rounds
        if finalized:
            break
        if declared_over:
            if not result["reconstructed"]:
                rejection_kind = "_reconstruction_failed"
            elif len(working) < minimum:
                rejection_kind = "_too_few"
            history.append({
                "speaker": "moderator_rejection", "cycle": cycle,
                "content": f"DEBATE_OVER rejected: reconstructed={result['reconstructed']}; questions={len(working)}; minimum={minimum}; cycle={cycle}; minimum_cycles={minimum_rounds}.",
            })
    if not working or not published:
        raise ModelResponseError(
            f"Pair {pair.pair_id}: no usable chairperson principles after {rounds} rounds. "
            "No constitution was fabricated; successful requests are checkpointed for retry."
        )
    history.append({"speaker": "debate_outcome", "cycles": cycle, "finalized": finalized, "reconstructed": result["reconstructed"]})
    return working, history
