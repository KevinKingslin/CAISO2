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

## Active DICAI source: supplied original code

On 2026-11-08 the user supplied `Democratic-ICAI-C8D5`. Its eight original files
are preserved byte for byte in `third_party/democratic_icai/original/`, under
MIT. `third_party/democratic_icai/PROMPTS.json` records every source hash and
all 15 complete request templates. `third_party/democratic_icai/prompts/`
contains the original requests; `caiso/prompts/dicai/` contains trajectory
versions. Original creative-writing persona constants are also preserved; CAISO
continues to use its configured dataset personas in debate.

The complete requests are independently reconstructed from the archived source
AST, without executing or importing that code. Tests compare all original
instructions and evidence layouts before checking the adapted files. The only
permitted text adaptations are:

1. Response/story terminology becomes trajectory terminology.
2. Human-choice wording becomes provided preference-label wording because CAISO
   obtains labels from the task metric.
3. Evidence fields contain CAISO's label-free structured execution records and
   use corresponding template placeholders. Gold, rewards and evaluation metadata
   remain hidden from the blind judge.
4. The five requests using the original reusable-criterion reasoning instruction
   append: "Where relevant, name the component IDs and describe observable
   behavior. Generalize question-specific entities and answer values into reusable
   criteria." This applies during CoT, sampling, consensus, drafting and revision;
   the original critique request is retained.
5. Debaters receive the previously agreed single sentence: "Proposed questions
   may concern individual component behavior, interactions between components,
   or the final output."
6. Abstraction receives: "Generalize example-specific entities and answer values
   while preserving relevant component IDs and the single quality shared by the
   principles."

Each principle starts with `Select the trajectory`. Its following grammar is
preserved: `that`, `where`, `which`, `with` and `in which` are all valid. Empty
criteria, placeholders and bare choices fail semantic validation. The constitution
remains a list of strings; no component-scope JSON schema or task-specific answer
style is introduced.

## Runtime alignment with those prompts

| Stage | Original behavior restored |
| --- | --- |
| CoT | One structured `reasoning` / `dimensions` / `questions` response |
| Self-consistency | Configured independent samples, recurring-dimension count and consensus request |
| Self-refinement | Draft followed by configured critique/revision iterations |
| Candidate questions | Taken from each rationale; named dimensions provide the original fallback if questions are absent |
| Debate | Three debaters then Moderator/Judge, one quality per question, reconstruction status and moderator rejection feedback |
| Finalization | Reconstruction must succeed, minimum question/round counts must hold, and the judge must declare `DEBATE_OVER` |
| Round cap | Original best-effort list retained, with `finalized: false` if gates fail |
| Inference | Original blind JSON `choice: "1" / "2"` plus reason, with shuffled trajectories |
| Abstraction | Original shared-rule prompt; singleton clusters retain their principle without a model request |
| Audit | Original three-axis prompt imported; no new audit stage or auditor model enabled |

Each normal and rejection-feedback request is a complete editable `.txt` file.
No new prompt fragments or hidden style suffixes are introduced. The old extra
CAISO seed-drafting prompt is removed because the original structured rationales
already supply their own questions. Original Stage-1 reasoning does not receive
an added persona preamble; the configured profiles enter the original debate
system prompts.

No existing numeric YAML parameter, benchmark prompt, dataset split, metric,
model route, clustering range or transport quota is changed. Previously dormant
reasoning and reconstruction settings now control their original source stages.
Each dimension is counted once per sampled chain, as in the source; tied counts
follow first-seen order for reproducible consensus inputs rather than Python's
process-dependent set iteration order.
This affects model-call counts and can affect the induced constitution. It is a
source-based trajectory adaptation, not a claim to reproduce every setting or
published score of DICAI. Report the adapted prompts and resolved experiment
settings in research.

The protocol identifier is `democratic-icai-c8d5-source-trajectory-v4-atomicity`.
Source, prompt and schema fingerprints invalidate earlier induction caches.
Start a new optimization run: old checkpoints retain constitutions generated
under their previous protocol. Successful requests within a matching induction
are still recovered after interruption, including independent samples with
identical messages and different seeds.

## CAISO one-quality semantic check

The 15 source-derived requests above are unchanged. Three complete CAISO-authored
requests in `caiso/prompts/method/constitution_atomicity/` enforce the existing
one-quality rule at runtime. Each changed chairperson list is reviewed for
independently satisfiable requirements. Compound items receive bounded split
revisions; unflagged items are preserved in Python. The repaired list is reviewed
again before reconstruction or publication. A repeated verified list needs no
new review. A failed revision retains the last verified working list; unverified
seed-only lists cannot become settled output.

Batched semantic review also checks all cluster abstractions. A compound
abstraction is revised to a single shared quality, keeping one output per source
cluster. If this fails, the failed clusters use existing, already checked medoid
principles; individually verified abstractions remain. This fallback and the
full review/revision traces are saved in constitution metadata. Clustering and
the silhouette-selected k are unchanged. Semantic checking is model-based and
fallible; it does not reject a principle merely for containing "and" or multiple
component IDs.

`dicai.atomicity.enabled: true`, `max_revisions: 2` and `batch_size: 20` are new
defaults. Bounded review/revision batches avoid oversized replies when the
constitution has many clusters. Setting
`enabled: false` restores format/reconstruction-only validation for ablations.
The existing induction model supplies all reviews and revisions, with ordinary
rate limits, request checkpoints and model-call accounting. There is no new
auditor deployment. Budget, service and content-filter errors retain their
existing behavior. The original three-axis bias/spuriousness/harmfulness audit
template remains unused. Tie labels and benchmark systems are unchanged.

This semantic guard is a documented CAISO extension to the source DICAI runtime,
not an original DICAI stage. Report the setting for research comparisons.

## Preserved history and runtime safeguards

The paper Appendix G source and `tests/fixtures/dicai_paper_prompts.json` remain
unmodified historical references. They were the active source before the user
supplied this checkout. The earlier anonymous-source audit and 51-request /
104-message consolidation capture remain historical metadata in
`PROMPT_AUDIT.json`; they do not describe the new active requests.

The new complete request capture is in
`tests/fixtures/complete_prompt_requests.json`. Benchmark prompts keep their
independent pinned-source equality checks. Missing resources are checked before
paid model calls; all active `.txt` files are packaged. Generic format repairs,
raw-error logging, rate limits, policy skips and validated request checkpoints
remain active. Invalid chairperson turns can retain the last validated list;
a list with no usable chairperson turn is never presented as settled. Auth,
budget and transport failures are not treated as format recovery.

```sh
python -m pytest -q tests/test_dicai_source_prompts.py tests/test_complete_prompts.py tests/test_config_and_prompts.py tests/test_benchmark_boundary.py
```
