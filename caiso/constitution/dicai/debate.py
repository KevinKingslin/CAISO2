"""Round-robin debate using arXiv:2606.28294v2, Appendix G.2 prompts."""

from caiso.core.recovery import ModelResponseError, RetryExhausted, response_failure
from caiso.prompts import render_messages

from .personas import Persona
from .preferences import pair_block
from .parsing import dedupe, explicitly_unchanged, parse_principles
from .parsing import principle_key as principle_key


def render(values):
    return "\n".join(f"- Q{i}: {value}" for i, value in enumerate(values, 1))


def debate(pair, catalog, providers, role, seed, rounds=4, config=None, progress=None):
    config = config or {}
    history_turns = config.get("history_turns", 8)
    temperatures = config.get("temperatures", {})
    if rounds < 1 or len(catalog) != 3:
        raise ValueError("DICAI requires three debaters and at least one round")
    working = dedupe([p for entry in catalog for p in entry.get("principles", [])])
    history = [{"speaker": "initial_working_list", "content": render(working)}]
    story = pair_block(pair, labeled=True)

    def recent():
        turns = [h for h in history if h["speaker"] != "initial_working_list"]
        return "\n\n".join(
            f"[{turn['speaker']}]\n{turn['content']}" for turn in turns[-history_turns:]
        ) or "(no debate history yet)"

    # The paper settles the list after T rounds; reconstruction is a later
    # diagnostic, not a stop-rejection prompt or a minimum-principle constraint.
    finalized = False
    published = False
    for cycle in range(1, rounds + 1):
        for i, entry in enumerate(catalog):
            if progress:
                progress(cycle, f"debater {i + 1}/3")
            persona = Persona.model_validate(entry["persona"])
            text = providers.complete(
                role,
                render_messages(
                    "dicai", "debate.debater",
                    agent_name=persona.identity, identity=persona.prompt(),
                    trace=entry["reasoning"],
                    # Own proposals are unnumbered, as in the original code.
                    # Only the shared working list assigns authoritative Q IDs.
                    seeds="\n".join(f"- {p}" for p in entry.get("principles", [])) or "(none)",
                    pair=story, working_list=render(working) or "(none)",
                    debate_history=recent(), cycle=cycle, total_cycles=rounds,
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
                role,
                render_messages(
                    "dicai", "debate.chairperson",
                    pair=story, working_list=render(working) or "(none)",
                    debate_history=recent(), cycle=cycle, total_cycles=rounds,
                ),
                seed=seed + (cycle - 1) * 4 + 3,
                validate=valid_update,
                temperature=temperatures.get("chairperson", 0.2),
                response_context={"pair_id": pair.pair_id, "stage": "chairperson", "cycle": cycle},
            )
        except (ModelResponseError, RetryExhausted) as error:
            if not response_failure(error):
                raise  # Auth, budget, transport failures and code errors remain fatal.
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
        finalized = "DEBATE_OVER" in chair.upper()
    if not working or not published:
        raise ModelResponseError(
            f"Pair {pair.pair_id}: no usable chairperson principles after {rounds} rounds. "
            "No constitution was fabricated; successful requests are checkpointed for retry."
        )
    history.append(
        {
            "speaker": "debate_outcome",
            "cycles": rounds,
            "finalized": finalized,
        }
    )
    return working, history
