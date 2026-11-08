# Democratic ICAI: end-to-end pipeline

Three stages, run in order:

1. **Generate** a constitution from preference data (multi-persona debate).
2. **Inference**: use the constitution to predict preferences, measure accuracy.
3. **Audit**: check each principle for bias, spurious criteria, and harmfulness.

```
preference_constitution/   # stage 1: induce the constitution
gpt_inference.py           # stage 2: constitution-guided preference prediction
audit/                     # stage 3: ethical audit of the principles
```

Each folder has its own detailed README; this is the connective overview.

---

## Data format

**Stage 1 (generation)** reads a CSV with these columns
(`preference_constitution/`):

| column | meaning |
|---|---|
| `task_prompt` | the shared prompt both responses answer |
| `chosen` | the human-preferred response |
| `rejected` | the other response |

Optional meta columns (`dataset`, `task`, `score_*`, ...) are carried through
if present. Which side (A/B) holds the chosen text is randomized per row with a
fixed seed, so the blind judge cannot exploit position.

**Stages 2 and 3 (inference / audit split)** use the ICAI-style schema:

| column | meaning |
|---|---|
| `input` | the prompt |
| `text_a`, `text_b` | the two responses |
| `preferred_text` | `text_a` or `text_b` (the human label) |

Keep a **train** split for stage 1 and a disjoint **test** split for stage 2.
Split on *response text*, not rows, to avoid leakage (see `audit/` for a
converter that does this).

---

## Credentials

Stages 1 and 2 use Azure OpenAI (GPT-4o / GPT-5) for generation and inference.
Stage 3 runs a local auditor model (Qwen or Llama) via `transformers`, with no
API. Put the Azure credentials in a `.env` file in the folder you run from:

```
AZURE_API_KEY=...
AZURE_API_BASE=https://<resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-10-21
```

The audit needs no key. Its models download from the HuggingFace Hub; the gated
Llama-3.1-70B requires a one-time `huggingface-cli login` (Qwen is ungated).

Deployment names are what you named your models in Azure, not model names; run
`preference_constitution/list_deployments.py` to list them.

---

## Stage 1: generate the constitution

```bash
cd preference_constitution
pip install -r requirements.txt

python run_pipeline.py \
  --data train.csv \
  --personas personas_research_questions.csv --dataset research_questions \
  --out out --minimize
```

For the full dataset run it in two phases (debate is expensive, clustering is
cheap and rerunnable):

```bash
python run_pipeline.py --data train.csv ... --out out --stages reason-debate --resume
python run_pipeline.py --data train.csv ... --out out --stages cluster --minimize
```

Output: `out/constitution.json` (the principles + reconstruction accuracy) and
`out/per_datapoint.jsonl` (per-pair checkpoint). See
`preference_constitution/README.md` for flags, cost, and resume behavior.

---

## Stage 2: inference (constitution-guided prediction)

Paste the induced principles into `gpt_inference.py` (the `constitution`
string), set the file paths, and run:

```bash
# in gpt_inference.py:
#   INPUT_CSV_FILE  = "test.csv"      # input/text_a/text_b/preferred_text
#   OUTPUT_CSV_FILE = "results.csv"
#   constitution    = """<paste principles here>"""

python gpt_inference.py
```

It asks the Azure judge to pick output (a) or (b) under the constitution for
each test pair, compares to `preferred_text`, and prints reconstruction
accuracy. Output CSV adds `LLM_response` and `correct` columns.

To turn `constitution.json` into the pasteable text block:

```bash
python -c "import json,sys; print(chr(10).join(p['principle'] for p in json.load(open('out/constitution.json'))['principles']))"
```

---

## Stage 3: audit the principles

The audit loads a local auditor model (a different family than the GPT models
that induced the constitution) via `transformers`; no API. Two are wired in:
`qwen` (Qwen2.5-32B-Instruct) and `llama` (Llama-3.1-70B-Instruct).

```bash
cd audit
pip install "transformers>=4.50" accelerate torch

python audit_principles.py \
  --input ../preference_constitution/out/constitution.json \
  --model qwen --out audit_out --name "my constitution"
```

`--input` accepts the `constitution.json` directly, or a `.txt` with one
principle per line. Each principle is flagged along three axes, bias / spurious
/ harmfulness, with a severity and rationale. Compare two constitutions
(ICAI vs DICAI) side by side with `audit_compare.py --model qwen`. A GPU is
required; see `audit/README.md` for VRAM, gated-model login, and the panel
workflow (run once per auditor and compare).

---

## Quick end-to-end

```bash
# 1. generate
cd preference_constitution
python run_pipeline.py --data train.csv --personas personas_research_questions.csv \
  --dataset research_questions --out out --minimize

# 2. inference (paste out/constitution.json principles into gpt_inference.py first)
cd ..
python gpt_inference.py

# 3. audit
cd audit
python audit_principles.py --input ../preference_constitution/out/constitution.json \
  --model qwen
```
