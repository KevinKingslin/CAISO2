"""Official round-robin debate and reconstruction gate, via CAISO transports.

Adapted from Democratic ICAI debate.py, MIT; third_party/democratic_icai.
"""

import re

from caiso.core.recovery import ModelResponseError
from caiso.prompts import entry as prompt_entry
from caiso.core.serialization import canonical
from caiso.providers.base import messages

from .personas import Persona
from .preferences import pair_block
from .reasoning_assembly import principle_instructions, prompt
from .reconstruction import reconstructs


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
    votes = config.get("reconstruction_votes", 1)
    minimum = config.get("min_principles", 3)
    min_cycles = config.get("min_debate_rounds", 1)
    history_turns = config.get("history_turns", 8)
    temperatures = config.get("temperatures", {})
    style = principle_instructions(config)
    if rounds < 1 or len(catalog) != 3:
        raise ValueError("DICAI requires three debaters and at least one round")
    working = dedupe([p for entry in catalog for p in entry["principles"]])
    history = [{"speaker": "initial_working_list", "content": render(working)}]
    rejection = ""
    story = pair_block(pair, labeled=True)

    def check(cycle, phase):
        result = reconstructs(
            providers, role, pair, working, votes, temperatures.get("reconstruction", 0.0)
        )
        history.append(
            {"speaker": "reconstruction", "cycle": cycle, "phase": phase, "result": result}
        )
        return result

    def recent(n):
        # Only agent turns enter the shared conversation, as in upstream.
        turns = [
            h for h in history if h["speaker"] not in {"initial_working_list", "reconstruction"}
        ]
        return canonical(turns[-n:])

    for cycle in range(1, rounds + 1):
        if progress:
            progress(cycle, "reconstruction before")
        result = check(cycle, "before")
        status = prompt_entry("method", "debate.reconstruction_status").format(
            status="YES" if result["reconstructed"] else "NO",
            predicted=result["predicted"], preferred=pair.preferred,
        )
        for i, entry in enumerate(catalog):
            if progress:
                progress(cycle, f"debater {i + 1}/3")
            persona = Persona.model_validate(entry["persona"])
            system = (
                prompt("debater").format(
                    name=persona.identity,
                    identity=persona.prompt(),
                    trace=entry["reasoning"],
                    seeds=render(entry["principles"]),
                )
                + "\n"
                + style
            )
            text = providers.complete(
                role,
                messages(
                    system,
                    canonical(
                        {
                            "pair": story,
                            "working_list": render(working),
                            "reconstruction": status,
                            "rejection": rejection,
                            "debate_history": recent(history_turns),
                            "cycle": cycle,
                        }
                    )
                    + prompt_entry('method', 'debate.guidance_1'),
                ),
                seed=seed + (cycle - 1) * 4 + i,
                temperature=temperatures.get("debater", 0.7),
            )
            history.append({"speaker": f"debater_{i + 1}", "content": text, "cycle": cycle})
        if progress:
            progress(cycle, "chairperson")
        chair = providers.complete(
            role,
            messages(
                prompt("chairperson").format(min_principles=minimum) + "\n" + style,
                canonical(
                    {
                        "pair": story,
                        "working_list": render(working),
                        "reconstruction": status,
                        "rejection": rejection,
                        "debater_arguments": recent(3),
                        "cycle": cycle,
                        "total_cycles": rounds,
                    }
                )
                + prompt_entry('method', 'debate.guidance_0').format(minimum=minimum),
            ),
            seed=seed + (cycle - 1) * 4 + 3,
            validate=lambda text: (parse_principles(text), text)[1],
            temperature=temperatures.get("chairperson", 0.2),
        )
        history.append({"speaker": "chairperson", "content": chair, "cycle": cycle})
        working = parse_principles(chair)
        if progress:
            progress(cycle, "reconstruction after")
        result = check(cycle, "after")
        over = "DEBATE_OVER" in chair.upper()
        if over and result["reconstructed"] and len(working) >= minimum and cycle >= min_cycles:
            break
        if over:
            reason = (
                prompt_entry("method", "debate.rejection_failed")
                if not result["reconstructed"]
                else prompt_entry("method", "debate.rejection_minimum").format(minimum=minimum, min_cycles=min_cycles)
            )
            rejection = prompt_entry("method", "debate.rejection").format(reason=reason)
            history.append({"speaker": "moderator", "content": rejection, "cycle": cycle})
    # Upstream returns best effort at the cycle cap. Never claim the gate passed.
    history.append(
        {
            "speaker": "debate_outcome",
            "cycles": cycle,
            "reconstructed": result["reconstructed"],
            "finalized": over
            and result["reconstructed"]
            and len(working) >= minimum
            and cycle >= min_cycles,
            "min_principles_met": len(working) >= minimum,
        }
    )
    return working, history
