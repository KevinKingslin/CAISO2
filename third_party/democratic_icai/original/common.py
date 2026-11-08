"""
common.py
Shared Azure OpenAI clients, LLM/embedding helpers, data structures, and
JSON parsing utilities used across the three pipeline stages.

Pipeline
    1. reasoning.py   rationales from CoT, CoT-SC, Self-Refine personas
    2. debate.py      per-datapoint principle set via AutoGen adversarial debate
    3. clustering.py  non-overlapping final constitution
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from openai import AzureOpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ------------------------------ configuration ----------------------------- #

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

# Every induced principle is a reusable preference rule of the form
#     "Select the response that <condition>."
# It is judged by reconstruction: applied to the pair, blind to the label, it
# must re-select the response the human actually preferred.
PRINCIPLE_STEM = "Select the response that"


# ------------------------------ azure client ------------------------------ #

_client: Optional[AzureOpenAI] = None


def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        if not AZURE_ENDPOINT or not AZURE_API_KEY:
            raise RuntimeError(
                "Missing Azure credentials. Set AZURE_OPENAI_ENDPOINT and "
                "AZURE_OPENAI_API_KEY (see .env.example)."
            )
        _client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
        )
    return _client


def _looks_reasoning(name: str) -> bool:
    """o1/o3/o4-style and GPT-5-style deployments reject temperature and
    max_tokens. Deployment names are user-chosen, so this is only a first
    guess; _SUPPORTS is corrected at runtime from the API's own errors.
    Set AZURE_OPENAI_REASONING=1 (or =0) to force it."""
    forced = os.environ.get("AZURE_OPENAI_REASONING", "").strip()
    if forced:
        return forced not in ("0", "false", "False", "no")
    n = (name or "").lower().lstrip("-_ ")
    return n.startswith(("o1", "o3", "o4", "o5")) or "gpt-5" in n or "gpt5" in n


# Corrected in place the first time the API rejects a parameter.
_SUPPORTS = {
    "temperature": not _looks_reasoning(CHAT_DEPLOYMENT),
    "max_completion_tokens": _looks_reasoning(CHAT_DEPLOYMENT),
    "response_format": True,
}

# reasoning models spend output budget on hidden reasoning tokens
_MIN_REASONING_TOKENS = 4000


def autogen_llm_config(temperature: float = 0.7) -> dict:
    """LLM config block for AutoGen agents, pointed at Azure OpenAI."""
    cfg = {
        "config_list": [
            {
                "model": CHAT_DEPLOYMENT,
                "api_type": "azure",
                "api_key": AZURE_API_KEY,
                "base_url": AZURE_ENDPOINT,
                "api_version": AZURE_API_VERSION,
            }
        ],
        "cache_seed": None,  # disable caching so each debate turn is fresh
        "timeout": 120,
    }
    if _SUPPORTS["temperature"]:
        cfg["temperature"] = temperature
    return cfg


# ------------------------------ llm helpers ------------------------------- #

def _chat_kwargs(messages, temperature: float, max_tokens: int,
                 json_mode: bool) -> dict:
    kwargs = {"model": CHAT_DEPLOYMENT, "messages": messages}
    if _SUPPORTS["temperature"]:
        kwargs["temperature"] = temperature
    if _SUPPORTS["max_completion_tokens"]:
        kwargs["max_completion_tokens"] = max(max_tokens, _MIN_REASONING_TOKENS)
    else:
        kwargs["max_tokens"] = max_tokens
    if json_mode and _SUPPORTS["response_format"]:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def _adapt(err: str) -> bool:
    """Turn an API parameter rejection into a capability update.
    Returns True if something changed and the call is worth retrying."""
    e = err.lower()
    if "temperature" in e and _SUPPORTS["temperature"]:
        _SUPPORTS["temperature"] = False
        return True
    if "max_tokens" in e and not _SUPPORTS["max_completion_tokens"]:
        _SUPPORTS["max_completion_tokens"] = True
        return True
    if "response_format" in e and _SUPPORTS["response_format"]:
        _SUPPORTS["response_format"] = False
        return True
    return False


def chat(messages, temperature: float = 0.7, max_tokens: int = 1200,
         json_mode: bool = False) -> str:
    for _ in range(4):
        try:
            resp = get_client().chat.completions.create(
                **_chat_kwargs(messages, temperature, max_tokens, json_mode))
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - re-raised unless it is a param issue
            if not _adapt(str(e)):
                raise
    raise RuntimeError("chat() could not find a supported parameter set")


def embed(texts: List[str]) -> List[List[float]]:
    resp = get_client().embeddings.create(model=EMBED_DEPLOYMENT, input=texts)
    return [d.embedding for d in resp.data]


# ------------------------------ json parsing ------------------------------ #

def extract_json(text: str) -> dict:
    """Recover the first JSON object from an LLM reply, tolerant of prose and
    ```json fences. Returns {} on failure rather than raising."""
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        if start == -1:
            return {}
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        break
    if candidate is None:
        return {}
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


# ------------------------------ data schema ------------------------------- #

@dataclass
class PreferencePair:
    id: str
    prompt: str
    response_a: str
    response_b: str
    preferred: str  # "A" or "B"
    meta: Dict = field(default_factory=dict)  # scores, dataset, task, etc.

    def preferred_text(self) -> str:
        return self.response_a if self.preferred.upper() == "A" else self.response_b

    def other_text(self) -> str:
        return self.response_b if self.preferred.upper() == "A" else self.response_a


@dataclass
class Rationale:
    datapoint_id: str
    strategy: str            # "cot" | "cot_sc" | "self_refine"
    preferred: str
    reasoning: str
    dimensions: List[str] = field(default_factory=list)
    # seed questions: [{"select_prompt": "Select the response that ...",
    #                   "reason": "why this criterion mattered"}]
    questions: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Principle:
    text: str
    source_datapoint: Optional[str] = None
    author: Optional[str] = None
    endorsements: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------- reconstruction judge ------------------------- #
# The core objective of the whole pipeline: a set of principles is only useful
# if applying it reconstructs the original preference. The judge below is BLIND
# to the gold label and sees the two responses in randomised order, so it cannot
# cheat. Reconstruction success = its choice equals the human's preference.

_RECON_SYS = (
    "You are a preference judge. You are given a set of principles and two "
    "responses to the same prompt. Decide which response the principles favour, "
    "using the principles as your ONLY criteria. You are not told which response "
    "a human preferred; judge solely from the principles."
)


def apply_constitution(pair: "PreferencePair",
                       principle_texts: List[str],
                       seed: Optional[int] = None) -> str:
    """Apply the principles to the pair, blind to the gold label and with the
    two responses shown in a randomised order. Returns the favoured side,
    'A' or 'B'."""
    rng = random.Random(seed)
    swap = rng.random() < 0.5
    first, second = ((pair.response_b, pair.response_a) if swap
                     else (pair.response_a, pair.response_b))
    principles = "\n".join(f"- {p}" for p in principle_texts) or "(no principles)"
    user = (
        f"PROMPT:\n{pair.prompt}\n\nPRINCIPLES:\n{principles}\n\n"
        f"RESPONSE 1:\n{first}\n\nRESPONSE 2:\n{second}\n\n"
        'Which response do the principles favour? Reply with JSON only: '
        '{"choice": "1" | "2", "reason": "<one sentence>"}.'
    )
    obj = extract_json(chat(
        [{"role": "system", "content": _RECON_SYS},
         {"role": "user", "content": user}],
        temperature=0.0, json_mode=True,
    ))
    picks_first = str(obj.get("choice", "1")).strip().startswith("1")
    if swap:
        return "B" if picks_first else "A"
    return "A" if picks_first else "B"


def reconstructs(pair: "PreferencePair",
                 principle_texts: List[str],
                 votes: int = 1) -> Tuple[bool, str]:
    """Whether the principles reconstruct the gold preference. With votes > 1,
    take a majority over order-randomised trials to reduce judge noise.
    Returns (success, predicted_side)."""
    if not principle_texts:
        return (False, "")
    tally = Counter(
        apply_constitution(pair, principle_texts, seed=i) for i in range(votes)
    )
    pred = tally.most_common(1)[0][0]
    return (pred == pair.preferred.upper(), pred)