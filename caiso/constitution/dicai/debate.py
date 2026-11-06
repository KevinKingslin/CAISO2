"""Round-robin debate using arXiv:2606.28294v2, Appendix G.2 prompts."""

import re

from caiso.core.recovery import ModelResponseError
from caiso.prompts import render_messages

from .personas import Persona
from .preferences import pair_block


def principle_key(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def dedupe(values):
    seen, output = set(), []
    for value in values:
        key = principle_key(value)
        if key and key not in seen:
            seen.add(key)
            output.append(value.strip())
    return output


def parse_principles(text):
    # Use the last final/working list, not Q-lines inside earlier quoted arguments.
    headers = list(re.finditer(r"(?:FINAL|WORKING)\s+LIST\s*:?", text, re.I))
    source = text[headers[-1].end() :] if headers else text
    values = re.findall(
        r"^\s*[-*]?\s*Q\s*\d+\s*[:.)]\s*(Select the response[^\n]+)", source, re.I | re.M
    )
    if not values:
        raise ModelResponseError("Chairperson produced no valid Q-numbered principles")
    return dedupe(values)


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
                    trace=entry["reasoning"], seeds=render(entry.get("principles", [])) or "(none)",
                    pair=story, working_list=render(working) or "(none)",
                    debate_history=recent(), cycle=cycle, total_cycles=rounds,
                ),
                seed=seed + (cycle - 1) * 4 + i,
                temperature=temperatures.get("debater", 0.7),
            )
            history.append({"speaker": f"debater_{i + 1}", "content": text, "cycle": cycle})
        if progress:
            progress(cycle, "chairperson")
        chair = providers.complete(
            role,
            render_messages(
                "dicai", "debate.chairperson",
                pair=story, working_list=render(working) or "(none)",
                debate_history=recent(), cycle=cycle, total_cycles=rounds,
            ),
            seed=seed + (cycle - 1) * 4 + 3,
            validate=lambda text: (parse_principles(text), text)[1],
            temperature=temperatures.get("chairperson", 0.2),
        )
        history.append({"speaker": "chairperson", "content": chair, "cycle": cycle})
        working = parse_principles(chair)
    history.append(
        {
            "speaker": "debate_outcome",
            "cycles": rounds,
            "finalized": "DEBATE_OVER" in chair.upper(),
        }
    )
    return working, history
