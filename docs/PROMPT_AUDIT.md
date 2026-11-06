# Prompt source and consolidation audit

Verified on 2026-11-06 against the sources below. This audit checks prompt text,
metadata and rendered messages; it does not establish reproduction of published
benchmark scores or equality of all runtime behavior.

## Sources

- OPTIMAS commit [`71871a2dd98081231d733debd41178ba47e13528`](https://github.com/snap-stanford/optimas/tree/71871a2dd98081231d733debd41178ba47e13528).
  The three original system files were checked against their SHA-256 values in
  `UPSTREAM.json` before extracting prompt literals and templates from their ASTs.
- [Anonymous Democratic-ICAI-C8D5 repository](https://anonymous.4open.science/r/Democratic-ICAI-C8D5/).
  All eight downloaded source files match the SHA-256 values already recorded in
  `third_party/democratic_icai/SOURCE.json`. No upstream Git commit is available.

## Benchmark system prompts: exact matches

| System | Original entries checked | Result |
| --- | ---: | --- |
| Amazon | 9 | Exact instruction, field metadata and rendered template matches |
| HotpotQA | 14 | Exact instruction and field metadata matches |
| PubMedQA | 6 | Exact instructions, scientist system message, answer format and rendered requests |

The 29 logical entries include four input templates, each rendered with three
sets of evidence. Checks include braces, quotes, non-ASCII text, blank lines,
empty candidate lists and role-marker-like text. PubMedQA checks compare complete
system/user messages; Amazon checks retain the original JSON serialization of
candidate lists. There are 41 individual equality checks when rendered cases are
counted separately.

The initial benchmark task prompts and static templates match OPTIMAS. Optimized
checkpoints can intentionally replace component instructions. Dataset-specific
constitution personas are separate CAISO inputs, not OPTIMAS benchmark prompts.

## DICAI prompts: restored from the arXiv paper

The active instruction text now follows [arXiv:2606.28294v2, Appendix G,
Figures 10–16](https://arxiv.org/abs/2606.28294v2), rather than the more elaborate
anonymous code snapshot. The original appendix LaTeX is retained in
[`third_party/democratic_icai/paper/appendix_g.tex`](../third_party/democratic_icai/paper/appendix_g.tex).
Tests compare every instruction word with that source, allowing only TeX layout,
list numbering and typography normalization.

- CoT, self-consistency and self-refine each use one persona-conditioned request
  and return only `final_reasoning`. The latter two strategies carry out their
  internal steps within that request, as specified by Figures 11–12. Since that
  schema contains no seed questions, the initial question list is empty;
  debaters propose questions through the paper's `ADD` section and the
  chairperson builds the working list. No additional seed-generation prompt is
  introduced.
- Chairperson and debater instructions match Figures 13–14. The added principle
  style, one-quality rule, reconstruction stop gate and minimum-principle
  instructions have been removed. All configured debate rounds run in order.
- Abstraction uses Figure 15. Blind constitution-guided inference uses Figure 16
  and parses `Output(a)` / `Output(b)`, retaining shuffled response order.
- The normal DICAI request receives no appended schema instruction. Validation,
  generic format-error retries, rate limiting and content-policy skips remain.

Complete requests still need dynamic inputs: CAISO supplies task/trajectory data,
persona profiles, reasoning traces, working lists, history and turn counters.
These are input data, not additional behavioral instructions from the paper.
The three QA/recommendation personas remain task adaptations. Existing model,
temperature, clustering and optional post-induction minimization settings remain
unchanged; this is not a claim of full experimental reproduction.

Legacy `self_consistency_samples`, `self_refine_steps`, `min_debate_rounds` and
`min_principles` settings no longer alter these paper requests. They remain in
existing configurations for compatibility. New prompt/protocol fingerprints
invalidate induction caches. Use a new optimization run to regenerate any
constitutions already saved in an old checkpoint.

## Historical comparison with the anonymous code snapshot

The pre-consolidation CAISO prompt components were compared directly with their
anonymous-source constants:

| Anonymous source constant | Finding in the CAISO port |
| --- | --- |
| `reasoning.py:_BASE_TASK` | Same wording; the text resource adds a final newline |
| `reasoning.py:_JSON_INSTRUCTION` | Same wording; the text resource adds a final newline |
| `common.py:_RECON_SYS` | Same wording; the text resource adds a final newline |
| `clustering.py:_ABSTRACT_SYS` | Same wording; the text resource adds a final newline |
| `debate.py:_DEBATER_SYS` | Modified: adds reusable-criterion guidance against sample-specific answer facts |
| `debate.py:_JUDGE_SYS` | Modified: strengthens reusable-principle constraints on reconstruction/finalization |
| `clustering.py:_ABSTRACT_USER` | Modified: asks for higher-level generalization, including singleton clusters, instead of the original lower-level abstraction wording |

CAISO also adds a shared reusable-principle style instruction to reasoning,
debate and abstraction. Its current request files contain that instruction in
full. Reasoning input spacing differs from the original source, and structured
CAISO task/trajectory evidence replaces original plain-text story comparisons in
several user messages. Method services add schema and response-repair messages.
The task-specific persona profiles are fictional CAISO adaptations, not the
original creative-writing persona dataset.

These findings describe the earlier CAISO prompts, before the explicit request
to restore the arXiv appendix. They are retained as migration history.

## Complete prompt files and regression checks

Stage request files hold `[SYSTEM]` and `[USER]` sections in one file. The
loader removes the role delimiters before formatting dynamic values. There are
no `guidance_0`/`guidance_1`, shared style suffixes or separate reasoning-output
instruction fragments to concatenate.

The complete pre-consolidation requests were captured before changing the caller
code: 51 requests, containing 104 message bodies. Their roles and UTF-8 hashes
are stored in `tests/fixtures/complete_prompt_requests.json`. The regression
capture exercises all reasoning stages, both debate rejection branches, blind
judging, abstraction, prompt optimization, block selection, workflow proposal,
schema guidance and format retries. This is a historical fixture: current tests
retain parity for its non-DICAI requests, while DICAI is checked against the
published paper source and `tests/fixtures/dicai_paper_prompts.json`.
These deterministic tests make no model API calls and do not measure task quality.

```sh
python -m pytest -q tests/test_complete_prompts.py tests/test_config_and_prompts.py tests/test_benchmark_boundary.py
```

Detailed original-source hashes and comparison results are in
[`PROMPT_AUDIT.json`](PROMPT_AUDIT.json).
