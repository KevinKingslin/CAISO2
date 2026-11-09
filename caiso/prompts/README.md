# Active prompt inventory

These files are read by CAISO and the configured benchmark systems. Benchmark
prompts retain pinned OPTIMAS text. Active DICAI prompts follow the supplied
original code in `third_party/democratic_icai/original/`, with minimal trajectory
adaptations. All requests are complete files; no prompt fragments or shared
instruction suffixes are assembled at runtime.

- `dicai/reasoning/cot.txt`: original CoT request.
- `dicai/reasoning/self_consistency.txt` and
  `self_consistency_consensus.txt`: independent samples followed by synthesis
  over recurring dimensions, using `self_consistency_samples`.
- `dicai/reasoning/self_refine.txt`, `self_refine_critique.txt` and
  `self_refine_revision.txt`: draft, critique and revision, using
  `self_refine_steps`.
- `dicai/debate/{debater,chairperson}.txt`: original participant and
  Moderator/Judge requests, with persona identity, own reasoning/questions,
  working lists, current-cycle arguments and reconstruction status.
- `dicai/debate/*_reconstruction_failed.txt` and `*_too_few.txt`: complete
  requests containing the original moderator rejection feedback. They repeat the
  full instructions intentionally so each file contains the entire request.
- `dicai/reconstruction.txt`: blind, shuffled trajectory judging, with original
  JSON `choice` (`"1"` / `"2"`) and `reason` output.
- `dicai/abstraction.txt`: original cluster abstraction; singleton clusters skip
  the model call and retain their principle.
- `dicai/audit.txt`: original external-auditor request, preserved as a template.
  Importing it does not enable a new audit stage or configure an auditor model.
- `method/<stage>/prompt.txt`: complete CAISO block-selection, prompt-search and
  workflow requests. Their existing tasks and schemas are unchanged.
- `method/services/schema_message.txt` and `method/providers/format_retry.txt`:
  complete fallback schema and response-repair messages.
- `method/preferences/*_pair.txt`: CAISO pair-evidence layouts.
- `method/constitution_atomicity/{check,split,abstract}.txt`: complete CAISO
  requests checking the original one-quality rule, splitting compound debate
  items, and revising compound cluster abstractions to one shared quality.
- `systems/<dataset>/<component>/`: unchanged benchmark task prompts/field metadata.
- `systems/<dataset>/personas/`: the configured three dataset-specific personas.

All generated principles begin with **`Select the trajectory`**. Parsing preserves
valid conditions such as `that`, `where`, `which`, `with` and `in which`; no exact
following connector is required. Empty criteria, bare A/B choices, trailing
ellipses, the existing condition/criterion placeholder forms and old response-style
prefixes remain invalid. Actual criteria and component identifiers are preserved.

The source instructions retain one-quality-per-question rules and reconstruction
feedback. Only response/story terminology, human-label wording and CAISO input
serialization are adapted, plus the documented reasoning, debate and abstraction
guidance. Reasoning considers discriminating intermediate-component behavior as
well as the final output. Every proposed question must name the exact trajectory
`component_id` values for the behavior it evaluates, including the producing
component for final-output questions. Reasoning questions and dimensions,
self-refinement critique, all debate variants, abstraction and existing repairs
explicitly exclude sample-specific entity names and values from generated rules.
They describe semantic roles while preserving actual component IDs and ask for
criteria that remain applicable when the example's entities change. Supporting
evidence retains the facts needed to compare trajectories. Chairpersons preserve
these IDs while editing or merging and reconstruct preferences through reusable
criteria. Output examples omit the literal component-ID placeholder;
abstraction requests the actual ID and reusable condition in natural language.
Split repair preserves the quality and logical conditions rather than requiring
instance-specific wording. These remain prompt instructions rather than an
entity-removal gate. Entity
removal and model compliance are not guaranteed: singleton clusters still copy
their settled principles and fallback medoids remain subject only to existing
checks. The source-adaptation audit documents the wording
changes within the 15 source-derived DICAI requests separately from the existing
atomicity repair prompts.
No dataset-specific answer styles or new structured constitution schema is added.

Component coverage is preserved across the complete induction set. Reasoners
and debaters inspect each component for observable differences explaining the
preference; chairpersons retain supported criteria for different components.
Abstraction retains exact IDs instead of aliases such as "final answer".
After abstraction and optional minimization, CAISO checks explicit ID coverage
against the active workflow (system scope) or selected block members. For each
missing component, it restores an existing settled debate principle whose source
pair executed the named components. No new model request or generic rule is
introduced. Components without an eligible settled principle remain listed as
insufficient evidence. This does not force every pair to cover every component.
The final reconstruction report judges the resulting list.

`component_coverage.json` and constitution metadata record expected/observed
components, coverage before/after restoration, missing components, restored
principles and their source pair IDs/settled indices. A latest system report is
saved directly in `.internal/dicai/system`; iteration system/block constitution
folders also receive it. Coverage measures explicit attribution, not semantic
correctness. Restored rules have the same checks as their settled debate source;
entity generalization and model-based atomicity remain fallible. See
[`docs/CONSTITUTION_COVERAGE.md`](../../docs/CONSTITUTION_COVERAGE.md).

CAISO additionally checks each new chairperson list for independently satisfiable
qualities. A bounded revision replaces only flagged items; unflagged principles
are preserved. A verified unchanged list reuses its earlier check. After cluster
abstraction, batched checks cover all output principles, including singleton
copies. Compound abstractions are revised to one shared quality per cluster.
If revisions fail, individually verified abstractions are retained and other
clusters use their already verified medoid principle. Failed debate revisions
retain the last verified list; a pair without any verified chairperson list
cannot produce settled principles. All judgments, revisions and fallbacks are
saved in pair transcripts or constitution metadata. These are fallible semantic
model judgments, not a rule based on conjunctions or sentence length.

The guard uses the existing induction model and request cache/budget. It is a
CAISO runtime extension, separate from the original three-axis audit template.
`dicai.atomicity.enabled` defaults to `true`; `max_revisions` defaults to `2`.
Reviews and revisions handle at most `batch_size: 20` principles per request to
keep large-constitution replies bounded. This is not a question batch size.
Disabling it restores the original format/reconstruction-only checks. This
changes induction calls and may change results; start a new run and report this
extension when describing the method.

Reasoning supplies questions directly in the original `reasoning`, `dimensions`,
`questions` JSON response. The former `reasoning/seed_questions.txt` helper is
removed. Original configured sample counts, refinement steps, minimum questions,
minimum rounds, temperatures and reconstruction votes now drive these source
stages; their numeric values are not changed. Persona profiles are used by the
original debate requests. The original Stage-1 reasoning instructions are kept
without an additional persona preamble.

Validated model requests are checkpointed under
`.internal/dicai/<scope>/<provenance>/requests/`, keyed by messages, seed, schema
and generation settings. Independent self-consistency samples intentionally have
identical messages with distinct seeds. Source/prompt changes invalidate older
induction caches. Start a new run to regenerate checkpointed constitutions.

Response validation, bounded repair retries, rate limits, policy skips and
per-request checkpoint recovery remain active. A malformed chairperson turn can
retain the last validated list and continue; a seed list with no usable
chairperson turn is never reported as settled. At the round cap the source's
best-effort list is retained with an explicit `finalized: false` outcome when the
reconstruction/finalization conditions were not met. Passing a minimum-question
gate never invents new criteria in Python.

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
Default optimizer tips match the preserved Optimas OPRO tips verbatim and in order,
in both the method YAML and fallback text files.
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

See [the original-source audit](../../docs/PROMPT_AUDIT.md). All 15 complete
original requests are preserved under `third_party/democratic_icai/prompts/`,
with source and resource hashes in `third_party/democratic_icai/PROMPTS.json`.
The supplied code and these derived requests retain its MIT license. The retained
historical paper source has its own CC BY-NC-SA 4.0 license.
