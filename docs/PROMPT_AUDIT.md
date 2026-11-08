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

## DICAI prompts: minimal trajectory adaptation of the arXiv paper

The active instruction text now follows [arXiv:2606.28294v2, Appendix G,
Figures 10–16](https://arxiv.org/abs/2606.28294v2), rather than the more elaborate
anonymous code snapshot. The original appendix LaTeX is retained in
[`third_party/democratic_icai/paper/appendix_g.tex`](../third_party/democratic_icai/paper/appendix_g.tex).
Tests compare the unmodified reference transcription with that source, allowing
only TeX layout, list numbering and typography normalization. Active requests
are checked against that transcription plus the following explicit adaptations:

- Reasoning evidence uses `TRAJECTORY A/B` and `PREFERRED TRAJECTORY` instead
  of response headings. Pair-evidence layouts use `TRAJECTORY A/B`; their
  supplied preference label is unchanged.
- Debaters receive trajectories and a provided preference label instead of
  stories and a human choice. Story/fact wording is updated accordingly. One
  sentence is added: "Proposed questions may concern individual component
  behavior, interactions between components, or the final output."
- Candidate drafting, debate and abstraction use the prefix
  `Select the trajectory where`. Parsing and format-repair feedback require the
  same prefix; old response-style output is not silently converted.
- Abstraction adds one requirement: "Preserve the component references and
  distinctions expressed by the input principles."
- Constitution-guided judging retains Figure 16's complete instructions,
  `Output(a)` / `Output(b)` labels, shuffled evidence and random fallback unchanged.

These edits do not prescribe dataset-specific answer styles or component
requirements. They allow the debate to express preferences grounded in the
existing execution records. The constitution remains a list of strings; no
structured component scopes or new debate stages are introduced. This is a
prompt adaptation and should be reported in experimental descriptions. The
protocol identifier is `democratic-icai-arxiv-2606.28294v2-trajectory-prompts-v2`.

- CoT, self-consistency and self-refine each use one persona-conditioned request
  and return only `final_reasoning`. The latter two strategies carry out their
  internal steps within that request, as specified by Figures 11–12. Their
  schema still contains no seed questions. A separate CAISO-authored extraction
  request (`dicai/reasoning/seed_questions.txt`) converts each final rationale
  into grounded candidate questions. The union initializes the working list,
  restoring the input supplied by `reasoning.py`/`seed_questions()` in the
  anonymous DICAI code. Starting from an empty list previously caused debaters
  to defend nonexistent questions and the chairperson to publish empty lists.
  This bridge is a documented implementation adaptation, not an eighth paper
  prompt transcription. It adds three budgeted, checkpointed calls per pair.
- Chairperson and debater instructions retain Figures 13–14 apart from the
  trajectory adaptations above. The earlier added principle-style block,
  one-quality rule, reconstruction stop gate and minimum-principle instructions
  remain removed. All configured debate rounds run in order.
- Abstraction adapts Figure 15 as listed above. Blind constitution-guided
  inference uses Figure 16 and parses `Output(a)` / `Output(b)`, retaining
  shuffled response order.
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
constitutions already saved in an old checkpoint. Existing saved constitutions
are not rewritten or migrated to the new wording automatically.

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

The earlier CAISO port also added a shared reusable-principle style instruction
to reasoning, debate and abstraction. Its request files contained that
instruction in full. Reasoning input spacing differs from the original source, and structured
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
remain available in commit `14f299c`. That historical capture exercised the old
reasoning stages, debate rejection branches and method requests.

CAISO's prompt optimizer, block selector and workflow designer now use the same
writing format as DICAI: named roles, numbered tasks, explicit JSON response
formats and labeled evidence sections. Their existing selection criteria,
workflow restrictions, inputs and output fields are retained. The formats are
included in each stage file; no duplicate schema message is appended to normal
stage requests. Generic schema and response-repair messages use the same style.
Preference layouts use trajectory headings; optimization tips, personas and
benchmark prompts retain their prior text. These CAISO templates are not
original DICAI paper prompts.

The current reviewed requests are recorded in
`tests/fixtures/complete_prompt_requests.json`. DICAI instruction wording is
also independently checked against the published paper source, the documented
runtime adaptation and
`tests/fixtures/dicai_paper_prompts.json`.
The paper reference retains its original text. The capture includes the
trajectory adaptation, candidate-question extraction requests and populated
seed/working-list inputs.
The response-repair request includes the actual invalid assistant answer before
the existing repair instruction. The current capture contains 41 requests and
85 message bodies.
These deterministic tests make no model API calls and do not measure task quality.

Runtime parsing accepts presentation variations without adding preference
criteria. Chairperson rounds can retain the last validated working list on a
format failure, with an explicit transcript event, and continue the configured
rounds. Error feedback no longer includes a copyable placeholder criterion.
An empty final list or a seed list with no usable chairperson turn remains an
error; seeds alone are not settled principles. Successful individual
induction requests are checkpointed and validated on reuse. These safeguards do
not add further normal-prompt instructions or change benchmark prompts/scoring.

```sh
python -m pytest -q tests/test_complete_prompts.py tests/test_config_and_prompts.py tests/test_benchmark_boundary.py
```

Detailed original-source hashes and comparison results are in
[`PROMPT_AUDIT.json`](PROMPT_AUDIT.json).
