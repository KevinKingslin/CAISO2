"""
debate.py  (Stage 2)

Adversarial debate over candidate preference questions, following the
Moderator/Judge and Debate-participant protocol.

Protocol
    Participants  Three persona debaters (seeded with their Stage-1 reasoning
                  trace and their own proposed question list). Each cycle they
                  DEFEND their strongest questions, ATTACK weak, redundant, or
                  vague ones, and may MERGE, EDIT, ADD, or CONCEDE. Questions
                  are always referenced by ID (Q1..Qn).
    Judge         Speaks only after all three debaters have spoken in a cycle.
                  Maintains the WORKING LIST, tracks support and objections,
                  and publishes an UPDATED WORKING LIST each cycle. On
                  finalization it publishes the FINAL LIST and prints
                  DEBATE_OVER.

Reconstruction gate
    The list is only useful if it reconstructs the human preference: a blind
    judge, shown the two stories in random order with no label and only these
    questions, must re-select the preferred story. Each cycle the reconstruction
    status of the current list is computed in code and given to every debater
    and to the Judge. The Judge MUST NOT finalize while the status is NO, and
    a hard guard rejects any DEBATE_OVER emitted against a failing list.

Note on the list: per the protocol, the Judge (not the debaters) mutates the
WORKING LIST. The list is therefore parsed from the Judge's published bullets.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import autogen

from common import (
    PreferencePair,
    Principle,
    Rationale,
    autogen_llm_config,
    reconstructs,
)

MAX_CYCLES = 4
MIN_CYCLES = 1
RECON_VOTES = 1        # majority over order-randomised reconstruction trials
MIN_PRINCIPLES = 3     # the judge may not finalize a list shorter than this
HISTORY_TURNS = 8      # recent transcript turns shown to each agent

# strategy -> (agent name, persona identity: style and values)
DEFAULT_PERSONAS: Dict[str, Tuple[str, str]] = {
    "cot": ("Literature_Professor",
            "a literature professor who values formal craft, constraint "
            "compliance, textual rigour, and clarity"),
    "cot_sc": ("Screenwriter",
               "a screenwriter who values voice, creative risk-taking, "
               "surprise, and concrete dramatic specificity"),
    "self_refine": ("Literary_Critic",
                    "a literary critic who values originality, world-building, "
                    "narrative energy, and freedom from generic tropes"),
}

_QLINE = re.compile(
    r"^\s*[-*]?\s*Q\s*(\d+)\s*[:.\)]\s*(Select the response[^\n]*)",
    re.IGNORECASE | re.MULTILINE,
)


# ------------------------------ working list ------------------------------ #

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        it = it.strip().strip('"').strip()
        key = _norm(it)
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def render_list(questions: List[str]) -> str:
    if not questions:
        return "(the working list is empty)"
    return "\n".join(f"- Q{i}: {q}" for i, q in enumerate(questions, 1))


def parse_working_list(text: str, fallback: List[str]) -> List[str]:
    """Read the Judge's published bullets. Prefer the block after the last
    'WORKING LIST' or 'FINAL LIST' header so quoted earlier lists are ignored."""
    if not text:
        return fallback
    upper = text.upper()
    anchor = max(upper.rfind("FINAL LIST"), upper.rfind("WORKING LIST"))
    segment = text[anchor:] if anchor != -1 else text
    items = [m.group(2) for m in _QLINE.finditer(segment)]
    if not items:
        items = [m.group(2) for m in _QLINE.finditer(text)]
    return _dedupe(items) or fallback


def seed_questions(rationales: List[Rationale]) -> List[str]:
    """Union of the personas' own proposed question lists, deduped."""
    out: List[str] = []
    for r in rationales:
        for q in r.questions:
            sp = (q.get("select_prompt") or "").strip()
            if sp:
                out.append(sp)
        if not r.questions:  # fall back to dimensions
            out.extend(
                f"Select the response that demonstrates higher {d}." for d in r.dimensions
            )
    return _dedupe(out)


# --------------------------- reconstruction gate -------------------------- #

def recon_status(pair: PreferencePair, questions: List[str],
                 votes: int = RECON_VOTES) -> Tuple[bool, str]:
    if not questions:
        return False, "RECONSTRUCTION STATUS: the working list is empty."
    ok, pred = reconstructs(pair, questions, votes=votes)
    return ok, (
        f"RECONSTRUCTION STATUS: applying ONLY these questions selects Story "
        f"{pred}; the human preferred Story {pair.preferred.upper()}; "
        f"reconstructed = {'YES' if ok else 'NO'}."
    )


# ------------------------------ agent prompts ----------------------------- #

_DEBATER_SYS = """You are {name}, a debater who must DEFEND your own proposed
questions and COUNTER other agents' questions.

Your persona identity (style/values): {identity}

You have access to:
- Story A
- Story B
- Preferred story text (human choice)
- Your persona identity (style/values)
- Your OWN persona reasoning trace (why the preferred story was chosen)
- Your OWN proposed question list (seed questions)
- The shared debate history, including other agents' proposals and the Judge's
  working list (Q1..Qn)

YOUR REASONING TRACE:
{trace}

YOUR SEED QUESTIONS:
{seeds}

DEBATE RULES (must follow):
- Always reference questions by ID (e.g., Q3, Q7) when defending or countering.
- Defend your strongest questions: explain why each criterion matters given your
  reasoning trace.
- Counter others: explain why a question is weaker, redundant, vague, or overlaps.
- You may propose merges: "Merge Q2 + Q5 into ..."
- You may concede: "Concede Q8 (too redundant)"
- Do not invent story facts beyond the given text.
- Keep arguments short, crisp, and comparative.

ONE QUALITY PER QUESTION (hard rule):
- Merge ONLY questions expressing the SAME quality in different words.
- Never fuse distinct qualities into one question. "Select the response that is
  specific, causal, and testable" is three questions wrongly fused into one.
- If two questions could be satisfied independently by a response, they are
  distinct and must stay separate, however tidy the merged version reads.

ARGUE FROM YOUR PERSONA, NOT FROM CONSENSUS:
- Your persona values a particular quality. Defend that quality above all others,
  and do not let it be merged away or deleted for the sake of a shorter list.
- Do not simply repeat another debater's merges. If another debater proposes
  folding your decisive quality into theirs, ATTACK that merge.
- Disagreement is the point. A cycle in which all three of you say the same thing
  has produced no information.

RECONSTRUCTION (the objective of the whole debate):
The final list must reconstruct the human preference. A blind judge, shown only
the questions and the two stories in random order with no label, must re-select
the preferred story. You are told the reconstruction status each cycle.
- A question that would select the NON-preferred story must be attacked or
  conceded, however reasonable it sounds in the abstract.
- A question true of BOTH stories discriminates nothing and must be attacked.
- If the status is NO, ADD the missing decisive question that carries the
  preference.

OUTPUT FORMAT (STRICT; no extra text):
1. DEFEND:
- Qx: <1-2 sentences why it should stay>

2. ATTACK:
- Qy: <1-2 sentences why it should be removed, merged, or edited>

3. MERGE:
- Merge Qm + Qn -> "Select the response..."

4. EDIT:
- Qk: "Select the response..."

5. ADD:
- "Select the response..." (only if truly missing and non-overlapping)

6. CONCEDE:
- Qz: <short reason>

Omit any section you do not use, but keep the numbering of those you do."""


_JUDGE_SYS = """You are the Moderator/Judge.

You MUST speak only after all three debaters have spoken in each cycle
(round-robin ensures this).

Your job:
1. Create and maintain a WORKING LIST of candidate questions labeled Q1..Qn.
2. Track support and objections for each Q based on debaters' DEFEND/ATTACK
   arguments.
3. After each cycle, publish an UPDATED WORKING LIST (bullets only), formatted as:

UPDATED WORKING LIST:
- Q1: Select the response ...
- Q2: Select the response ...

4. When updating:
- Remove questions with strong objections and little defense.
- Merge overlapping questions if proposed (or if obvious).
- Apply edits if they improve clarity.
- Ensure every item begins exactly with "Select the response"
- Keep the list concise and non-overlapping.

ONE QUALITY PER QUESTION (hard rule):
Each question names exactly ONE quality. Never conjoin distinct qualities into a
single question. "Select the response that is specific, causal, and testable" is
THREE questions wrongly fused into one, and is forbidden. Merge only questions
that express the SAME quality in different words. Two questions that a response
could satisfy independently are distinct qualities and must stay separate.
Never delete a quality that a debater is still actively defending just to make
the list shorter.

RECONSTRUCTION CHECK (mandatory, overrides everything else):
Each cycle you are given the reconstruction status of the current list: whether
a blind judge, applying only these questions to the two stories shown in random
order with no label, re-selects the human-preferred story.
- If the status is NO, you MUST NOT finalize. Remove any question that would
  select the non-preferred story, remove questions true of both stories, and
  retain or add the question that carries the preference.
- You may finalize ONLY when the status is YES, the list is stable (few changes
  this cycle), and the list is concise and non-overlapping.
- The list must retain at least {min_principles} distinct questions. A shorter
  list means you have fused distinct qualities; split them apart again.

Finalization:
After the final cycle, publish FINAL LIST (bullets Q1..Qn) and then print:

DEBATE_OVER"""


# ------------------------------ agent plumbing ---------------------------- #

def _speak(agent: autogen.ConversableAgent,
           moderator: autogen.ConversableAgent,
           content: str) -> str:
    out = agent.generate_reply(
        messages=[{"role": "user", "content": content}], sender=moderator
    )
    if isinstance(out, dict):
        out = out.get("content", "")
    return (out or "").strip()


def _story_block(pair: PreferencePair) -> str:
    return (
        f"TASK PROMPT:\n{pair.prompt}\n\n"
        f"STORY A:\n{pair.response_a}\n\n"
        f"STORY B:\n{pair.response_b}\n\n"
        f"PREFERRED STORY (human choice): Story {pair.preferred.upper()}\n"
        f"PREFERRED STORY TEXT:\n{pair.preferred_text()}"
    )


def _history(transcript: List[str], n: int = HISTORY_TURNS) -> str:
    if not transcript:
        return "(no debate history yet)"
    return "\n\n".join(transcript[-n:])


# ------------------------------ debate driver ----------------------------- #

def run_debate(pair: PreferencePair,
               rationales: List[Rationale],
               max_cycles: int = MAX_CYCLES,
               votes: int = RECON_VOTES,
               personas: Optional[Dict[str, Tuple[str, str]]] = None,
               transcript_path: Optional[str] = None,
               min_principles: int = MIN_PRINCIPLES,
               verbose: bool = True) -> List[Principle]:
    """Run the adversarial debate for one datapoint. Returns the Judge's final
    list, guaranteed (where achievable) to reconstruct the human preference.
    If transcript_path is given the full debate history is written there, even
    if the debate raises."""

    personas = personas or DEFAULT_PERSONAS
    by_strategy = {r.strategy: r for r in rationales}
    transcript: List[str] = []

    try:
        return _debate(pair, by_strategy, personas, transcript,
                       max_cycles, votes, min_principles, verbose)
    finally:
        if transcript_path:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(f"DATAPOINT {pair.id}\n{_story_block(pair)}\n\n")
                f.write("\n\n".join(transcript))


def _debate(pair, by_strategy, personas, transcript,
            max_cycles, votes, min_principles, verbose) -> List[Principle]:

    moderator = autogen.UserProxyAgent(
        name="Moderator_Proxy", human_input_mode="NEVER",
        code_execution_config=False, llm_config=False, default_auto_reply="",
    )

    debaters: List[autogen.AssistantAgent] = []
    for strat, (name, identity) in personas.items():
        r = by_strategy.get(strat)
        own_seeds = seed_questions([r]) if r else []
        debaters.append(autogen.AssistantAgent(
            name=name,
            system_message=_DEBATER_SYS.format(
                name=name, identity=identity,
                trace=(r.reasoning if r else "(none)"),
                seeds="\n".join(f"- {s}" for s in own_seeds) or "(none)",
            ),
            llm_config=autogen_llm_config(temperature=0.7),
        ))

    judge = autogen.AssistantAgent(
        name="Judge",
        system_message=_JUDGE_SYS.format(min_principles=min_principles),
        llm_config=autogen_llm_config(temperature=0.2),
    )

    stories = _story_block(pair)
    rationales = list(by_strategy.values())
    questions = seed_questions(rationales)   # seed WORKING LIST

    if verbose:
        print("[judge] SEED WORKING LIST")
        print(render_list(questions))

    rejection = ""   # the moderator's last DEBATE_OVER rejection, if any

    for cycle in range(1, max_cycles + 1):
        ok, status = recon_status(pair, questions, votes)
        if verbose:
            print(f"\n=== CYCLE {cycle} ===\n[status] {status}")

        # -- all three debaters speak, round-robin -------------------------
        for d in debaters:
            prompt = (
                f"{stories}\n\nJUDGE'S CURRENT WORKING LIST:\n"
                f"{render_list(questions)}\n\n{status}\n"
                f"{rejection}\n"
                f"SHARED DEBATE HISTORY:\n{_history(transcript)}\n\n"
                f"Cycle {cycle}. Defend, attack, merge, edit, add, or concede. "
                "Follow the STRICT output format."
            )
            reply = _speak(d, moderator, prompt)
            transcript.append(f"[{d.name}]\n{reply}")
            if verbose:
                print(f"\n--- {d.name} ---\n{reply}")

        # -- judge speaks last, updates the list ---------------------------
        j_prompt = (
            f"{stories}\n\nCURRENT WORKING LIST:\n{render_list(questions)}\n\n"
            f"{status}\n{rejection}\n"
            f"DEBATER ARGUMENTS THIS CYCLE:\n"
            f"{_history(transcript, n=len(debaters))}\n\n"
            f"Cycle {cycle} of {max_cycles}. Publish the UPDATED WORKING LIST. "
            f"Finalize only if the reconstruction status is YES, the list is "
            f"stable and non-overlapping, and it retains at least "
            f"{min_principles} distinct one-quality questions."
        )
        j_reply = _speak(judge, moderator, j_prompt)
        transcript.append(f"[Judge]\n{j_reply}")
        if verbose:
            print(f"\n--- Judge ---\n{j_reply}")

        previous = questions
        questions = parse_working_list(j_reply, fallback=previous)
        declared_over = "DEBATE_OVER" in j_reply.upper()

        # -- hard gate: reconstruction must hold on the NEW list -----------
        ok, status = recon_status(pair, questions, votes)
        too_few = len(questions) < min_principles
        if declared_over and ok and not too_few and cycle >= MIN_CYCLES:
            if verbose:
                print(f"\n[gate] {status}\n[judge] DEBATE_OVER accepted.")
            break
        if declared_over and not ok:
            note = ("[Moderator] DEBATE_OVER REJECTED: the published list does "
                    "not reconstruct the human preference. Continue the debate: "
                    "remove questions that select the non-preferred story and "
                    "add the decisive one.")
            transcript.append(note)
            rejection = note
            if verbose:
                print(f"\n[gate] {status}\n{note}")
        elif declared_over and too_few:
            note = (f"[Moderator] DEBATE_OVER REJECTED: the list has "
                    f"{len(questions)} question(s), fewer than the required "
                    f"{min_principles}. You have fused distinct qualities into "
                    f"one compound question. Split them apart: each question must "
                    f"name exactly ONE quality, and no quality still being "
                    f"defended may be deleted merely to shorten the list.")
            transcript.append(note)
            rejection = note
            if verbose:
                print(f"\n[gate] {note}")
    else:
        if verbose and not recon_status(pair, questions, votes)[0]:
            print("\n[warn] cycles exhausted without reconstruction; "
                  "returning best-effort list.")

    if verbose:
        print("\nFINAL LIST")
        print(render_list(questions))

    return [Principle(text=q, source_datapoint=pair.id, author="Judge")
            for q in questions]


if __name__ == "__main__":
    from reasoning import run_reasoning

    demo = PreferencePair(
        id="story-001",
        prompt=("Come up with an original and creative story which includes the "
                "following 3 words, make it around 5 sentences long: statement, "
                "stealth, detect."),
        response_a=("Today in the paper the headline read, \"Sloths are taking "
                    "over Manhattan\". The statement came straight from the mayor "
                    "himself. The mayor said that due to the stealth and sneakiness "
                    "of the sloths, the police can't detect where they are coming "
                    "from. You can find them all over the city just hanging out. "
                    "They are hanging on buildings riding the publi"),
        response_b=("The career politician made a careful statement to the press. "
                    "He thoughtfully chose his words, making sure no reporter could "
                    "detect the crime he was trying to hide. It had taken "
                    "considerable stealth and planning to pull off the heist and he "
                    "wanted to deflect any suspicion away from himself and his staff. "
                    "This wasn't the first time the politician had broken the law but "
                    "this time it was different."),
        preferred="A",
    )
    rats = run_reasoning(demo)
    for p in run_debate(demo, rats):
        print(" -", p.text)