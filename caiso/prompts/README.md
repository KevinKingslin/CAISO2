# Active prompt inventory

These files are read by CAISO and the three configured benchmark systems.
Benchmark prompts retain OPTIMAS text; DICAI uses the arXiv paper instructions.

- `dicai/reasoning/{cot,self_consistency,self_refine}.txt`: complete requests
  matching Appendix G, Figures 10–12 of arXiv:2606.28294v2. Each returns only
  `final_reasoning`; self-consistency/refinement happen within that one request.
- `dicai/debate/debater.txt` and `chairperson.txt`: complete system/user requests,
  matching Figures 13–14, plus dynamic persona/evidence/history inputs.
- `dicai/reconstruction.txt` and `abstraction.txt`: complete requests for blind
  judging and cluster abstraction.
- `method/<stage>/prompt.txt`: complete block-selection, prompt-optimization and
  workflow-proposal requests written in DICAI's format: a named agent, available
  evidence, numbered tasks, requirements and an explicit JSON output format.
  User sections supply evidence under descriptive headings. These are CAISO
  prompts, not transcriptions of the DICAI paper.
- `method/services/schema_message.txt` and `method/providers/format_retry.txt`:
  complete fallback schema and response-repair messages in the same style.
- `method/preferences/*_pair.txt`: complete labeled/unlabeled evidence layouts.
- `systems/hotpotqa/<component>/`: DSPy instructions and field prefixes.
- `systems/amazon/<component>/`: both Qwen completion templates and the DSPy next-item
  instruction, prefixes and field descriptions.
- `systems/pubmedqa/`: scientist system message, component instructions,
  yes/no/maybe format and context-analysis/problem-solving user templates.

Each request template contains the complete instructions for its stage. The
paper's output-format instructions and debate guidance are written directly into
the corresponding request. No extra principle style, reconstruction stop gate,
minimum-principle rule or moderator rejection feedback is added to DICAI.
For example, open `dicai/debate/chairperson.txt` to see its system instructions,
evidence fields, and instruction to publish the updated working list together.

The three CAISO stage requests already include their required response formats,
so the transport does not append a duplicate schema instruction to them. JSON
validation and generic response-repair retries remain active. Preference-pair
layouts, configured optimization tips and persona profiles retain their text.

Chat request files use role sections:

```text
[SYSTEM]
Complete system instructions for this stage.
[USER]
Dynamic evidence: {evidence:json}
Complete instructions for this turn.
```

The headers are file delimiters, not text sent to the model. Placeholders such as
`{identity}`, `{preferred}` and `{history:json}` hold dynamic values. The `:json`
specifier serializes evidence canonically; literal JSON braces in a template are
escaped as `{{` and `}}`. Inputs are substituted after role parsing, so role-like
text inside evidence stays data. Message whitespace is preserved exactly.

Amazon Qwen `prompt.txt` files are complete raw-completion prompts and have no
role delimiters. PubMedQA component `prompt.txt` files contain both the scientist
system message and the complete user template, including the solver's final-answer
format. `instructions.txt` holds each component's initial optimizable instruction.
For DSPy components, field prefixes/descriptions in `fields/` are metadata; DSPy's
runtime generates its chat-format wrappers around the component instruction.

Each system also has `personas/` with three numbered `.txt` profiles restored from
the previous CAISO configuration. These are fictional task adaptations with the
full attribute descriptions, rather than the DICAI paper's creative-writing
personas. The rendered profile text is preserved exactly.

| Dataset | Persona roles, in file order |
| --- | --- |
| Amazon | E-commerce behavior analyst; product matching specialist; recommendation evaluator |
| HotpotQA | Multi-hop question answering researcher; information retrieval specialist; fact verification specialist |
| PubMedQA | Biomedical evidence analyst; clinical research methodologist; scientific question answering evaluator |

```text
systems/
  amazon/
    session_analyzer/prompt.txt
    candidate_profiler/prompt.txt
    next_item_decider/instructions.txt
    next_item_decider/fields/*.txt
    personas/*.txt
  hotpotqa/
    question_rewriter/{instructions.txt,fields/}
    info_extractor/{instructions.txt,fields/}
    hint_generator/{instructions.txt,fields/}
    answer_generator/{instructions.txt,fields/}
    personas/*.txt
  pubmedqa/
    context_analyst/{instructions.txt,prompt.txt}
    problem_solver/{instructions.txt,prompt.txt}
    personas/*.txt
```

Ordered default tips and personas use one numbered `.txt` file per entry under
`method/prompt_optimizer/default_tips/` and `method/default_personas/`.
`_catalogs.py` maps existing caller keys to these resources and keeps list order;
it contains paths, not prompt content. The loaders read only the requested text
files and preserve all whitespace, including trailing spaces in field prefixes.
Tips, personas and initial instructions are variable inputs to complete requests,
not hidden instruction suffixes.

Placeholders are filled by the caller; input containing braces is not interpreted
as another template. DSPy generates its own chat-format wrappers. Schemas,
trajectory/context data and candidate instructions are generated at runtime;
newly optimized prompts remain in run artifacts.

The experiment's `caiso.persona_catalog` selects its three dataset profiles for
both system and block induction. `prompt_tips` and custom `caiso.personas`
overrides remain configuration parameters; explicit lists take precedence over
catalogs. The generic `method/default_personas/` profiles remain a fallback for
systems outside the three configured datasets. Optional preserved
STaRK/BigCodeBench examples and legacy OPTIMAS utilities retain their original prompt
code; they are not part of the three active experiment profiles.

DICAI includes its templates and the labeled-pair/response-repair templates it
uses in cache provenance, along with its resolved personas and configuration.
Unused generic personas, optimizer tips and disabled workflow templates are not
loaded by constitution induction. Prompt changes can affect optimization results;
start a new run to regenerate constitutions saved in older checkpoints.

Check every catalog resource without making model calls:

```sh
python -m scripts.verify_prompts
```

Optimization checks its required resources before loading benchmark models and
reports all missing paths together. Required files remain mandatory; configured
personas and tips do not depend on the unused default files. Clean wheels and
source distributions include all catalog `.txt` files.

For an incomplete installation, update the complete checkout and reinstall with
`python -m pip install --no-cache-dir --force-reinstall --no-deps -e .`.
If a listed tracked file was deleted locally, restore that missing file from
Git; preserve any other prompt edits. Check the resource report's package path
to confirm that Python is loading this checkout.

See [the original-source audit](../../docs/PROMPT_AUDIT.md): benchmark system
prompts match pinned OPTIMAS; DICAI instruction wording is checked against the
paper's retained LaTeX source, with dynamic task evidence and dataset personas
documented separately. The prompt transcriptions retain the paper's CC BY-NC-SA
4.0 license and attribution in `third_party/democratic_icai/paper/`.
