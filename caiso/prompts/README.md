# Active prompt inventory

These files are read by CAISO and the three configured benchmark systems. Text,
whitespace and configuration/generation values were preserved during extraction.

- `dicai/*.txt`: the eight DICAI reasoning, style, debate, blind judging and
  abstraction templates, moved from `constitution/dicai/prompts`.
- `method.json`: reasoning-stage and debate guidance, preference formatting, block
  selection, prompt optimization, workflow proposal, default personas, schema
  instructions and response repair.
- `systems/hotpotqa.json`: DSPy instructions and field prefixes.
- `systems/amazon.json`: both Qwen completion templates and the DSPy next-item
  instruction, prefixes and field descriptions.
- `systems/pubmed.json`: scientist system message, component instructions,
  yes/no/maybe format and context-analysis/problem-solving user templates.

JSON escapes such as `\n` represent newlines. Placeholders are filled by the caller;
input containing braces is not interpreted as another template. DSPy generates its
own chat-format wrappers. Schemas, trajectory/context data and candidate instructions
are generated at runtime; newly optimized prompts remain in run artifacts.

`prompt_tips` and custom persona overrides remain configuration parameters. The
defaults are visible here; explicit config values take precedence. Optional preserved
STaRK/BigCodeBench examples and legacy OPTIMAS utilities retain their original prompt
code; they are not part of the three active experiment profiles.

DICAI includes method fragments and its text templates in cache provenance, so
editing them invalidates induction caches. Prompt changes can affect benchmark
scores; this extraction retains the previous content.
