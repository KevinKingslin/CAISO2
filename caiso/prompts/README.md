# Active prompt inventory

These files are read by CAISO and the three configured benchmark systems. Text,
whitespace and configuration/generation values were preserved during extraction.

- `dicai/*.txt`: the eight DICAI reasoning, style, debate, blind judging and
  abstraction templates, moved from `constitution/dicai/prompts`.
- `method/`: reasoning-stage and debate guidance, preference formatting, block
  selection, prompt optimization, workflow proposal, default personas, schema
  instructions and response repair.
- `systems/hotpotqa/<component>/`: DSPy instructions and field prefixes.
- `systems/amazon/<component>/`: both Qwen completion templates and the DSPy next-item
  instruction, prefixes and field descriptions.
- `systems/pubmedqa/`: scientist system message, component instructions,
  yes/no/maybe format and context-analysis/problem-solving user templates.

Every active prompt is a `.txt` resource with ordinary, editable newlines. Component
instructions live in `instructions.txt`; complete Qwen prompts in `prompt.txt`;
PubMedQA user templates in `user.txt`; DSPy field prefixes and descriptions in each
component's `fields/` folder. PubMedQA's shared scientist message is `system.txt`, and
its final-answer format is `problem_solver/answer_format.txt`.

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
    system.txt
    context_analyst/{instructions.txt,user.txt}
    problem_solver/{instructions.txt,user.txt,answer_format.txt}
    personas/*.txt
```

Ordered default tips and personas use one numbered `.txt` file per entry under
`method/prompt_optimizer/default_tips/` and `method/default_personas/`.
`_catalogs.py` maps existing caller keys to these resources and keeps list order;
it contains paths, not prompt content. The loaders read only the requested text
files and preserve all whitespace, including trailing spaces in field prefixes.

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

DICAI includes method fragments and its text templates in cache provenance, so
editing them invalidates induction caches. Prompt changes can affect benchmark
scores; this extraction retains the previous content.
