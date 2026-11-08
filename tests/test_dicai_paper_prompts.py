"""Retain the historical paper transcription; active prompts use the supplied code."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = json.loads((ROOT / "tests/fixtures/dicai_paper_prompts.json").read_text())


def paper_words(value):
    """Ignore TeX layout, typography and list numbering, retaining every word."""
    value = re.sub(r"\\vspace\{[^}]*\}", "", value)
    value = re.sub(r"\\\\\[[^]]*\]", "", value)
    value = re.sub(r"\\(?:begin|end)\{[^}]*\}(?:\[[^]]*\])?", "", value)
    value = re.sub(r"\\item(?:\[[^]]*\])?", "", value)
    value = re.sub(r"\\(?:noindent|texttt|textbf)", "", value)
    value = re.sub(r"\\textless(?:\{\})?", "<", value)
    value = re.sub(r"\\textgreater(?:\{\})?", ">", value)
    value = value.replace(r"\_", "_").replace("’", "'")
    value = re.sub(r"^\s*\d+\.\s*", "", value, flags=re.M)
    return re.findall(r"[\w]+(?:'[\w]+)?", value)


def test_transcriptions_match_published_latex_words():
    source = (ROOT / "third_party/democratic_icai/paper/appendix_g.tex").read_text()
    blocks = re.findall(r"\\begin\{tcolorbox\}\[.*?\]\n(.*?)\\end\{tcolorbox\}", source, re.S)
    keys = [
        "reasoning.cot", "reasoning.self_consistency", "reasoning.self_refine",
        "debate.chairperson", "debate.debater", "abstraction", "reconstruction",
    ]
    for key, block in zip(keys, blocks, strict=False):
        expected = PAPER["instructions"][key]
        if isinstance(expected, dict):
            expected = "System: " + expected["system"] + " User: " + expected["user"]
        assert paper_words(block) == paper_words(expected), key
    assert len(blocks) >= len(keys)
