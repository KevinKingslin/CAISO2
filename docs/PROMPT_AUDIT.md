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
4. The five reusable-criterion reasoning requests consider discriminating
   intermediate-component behavior as well as final outputs. Every question
   names the responsible components using actual execution IDs, including the
   final-output producer. The generalization rule below applies to proposed
   questions and dimensions during CoT, sampling, consensus, drafting and revision.
5. The self-refinement critique additionally flags sample-specific names or facts
   used as criteria or dimensions and applies the same generalization rule.
6. All three debater variants allow individual-component, interaction and
   final-output criteria with actual component IDs. They include the
   generalization rule and reconstruct the preference through reusable criteria.
7. All three chairperson variants preserve actual component IDs when editing or
   merging and include the generalization rule. Reconstruction remains mandatory;
   its heading no longer says it overrides every other instruction. It explicitly
   uses reusable component-behavior criteria. Its reconstruction, round and
   minimum-question runtime gates remain unchanged.
8. Abstraction preserves one shared quality, applies the generalization rule and
   names evaluated components using actual IDs from the member principles.
9. Reasoning and chairperson examples retain the trajectory prefix without a
   literal component-ID token. Abstraction describes the actual ID and condition
   in natural language. The `<component_id>` token was removed after a generated
   principle copied it literally.

The generalization instruction is written inside each complete request:

```text
Express the reusable quality that explains the preference.
Exclude sample-specific names of people, places, organizations and works,
and example-specific dates and identifiers from principles.
Describe their semantic roles instead, while preserving actual component IDs.
The principle should remain applicable when those entities change.
```

The reasoning and critique versions scope the exclusion to proposed questions
or criteria and dimensions. The instruction concerns generated rules; raw
trajectories and supporting comparison evidence retain their facts. Criteria
remain grounded in the paired executions, with no per-pair component quota.
The final constitution must cover every target component with a separate
subject-attributed principle. The debate structure, reconstruction objective and one-quality rule remain intact.

This prompt-only revision was applied on 2026-11-09 following observed
sample-specific principles. It adds no validation gate, retry policy, model route
or call stage. Prompt compliance, entity removal and correct attribution are not
guaranteed by the runtime checks. Every cluster, including singletons, receives
a model abstraction request. Failed abstraction repairs can still use an atomic
medoid, so producing reusable rules during debate remains essential.

As of 2026-11-10, every principle must use `Select the trajectory where`, followed
immediately by an actual component ID and one observable, reusable criterion.
Equivalent connectors now trigger format repair. Capitalization and whitespace
are normalized; IDs and criteria retain their spelling. Component subjects are
validated against execution IDs in reasoning, debate, abstraction and semantic
revision. Actual reasoning questions are required; dimensions alone no longer
produce generic fallback criteria. Complete generating and repair requests
state this format and the final per-component coverage requirement without a
literal component-ID placeholder. Constitutions remain lists of strings.

## Runtime alignment with those prompts

| Stage | Original behavior restored |
| --- | --- |
| CoT | One structured `reasoning` / `dimensions` / `questions` response |
| Self-consistency | Configured independent samples, recurring-dimension count and consensus request |
| Self-refinement | Draft followed by configured critique/revision iterations |
| Candidate questions | Taken from each rationale; actual component-attributed questions are required |
| Debate | Three debaters then Moderator/Judge, one quality per question, reconstruction status and moderator rejection feedback |
| Finalization | Reconstruction must succeed, minimum question/round counts must hold, and the judge must declare `DEBATE_OVER` |
| Round cap | Original best-effort list retained, with `finalized: false` if gates fail |
| Inference | Original blind JSON `choice: "1" / "2"` plus reason, with shuffled trajectories |
| Abstraction | Shared-rule prompt applied to every cluster, including singleton clusters |
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

The protocol identifier is `democratic-icai-c8d5-source-trajectory-v6-required-component-principles`.
Source, prompt and schema fingerprints invalidate earlier induction caches.
Start a new optimization run: old checkpoints retain constitutions generated
under their previous protocol. Successful requests within a matching induction
are still recovered after interruption, including independent samples with
identical messages and different seeds.

## Component coverage extension

Reasoning and debate requests now explicitly inspect every component for
observable differences explaining the preference. The critique flags overlooked
component differences, chairpersons preserve each component's supported
criteria when shortening lists, and abstraction keeps exact IDs instead of
generic references to a final answer or rewritten query. These are short edits
within the existing complete source-derived requests. No per-pair component
quota is imposed, and reconstruction/audit request text is unchanged.

After abstraction and optional minimization, CAISO restores settled debate rules
for missing workflow/block component IDs when their source trajectories executed
those components. Coverage counts only the exact component ID immediately after the required
prefix, so an interaction mentioning another component does not cover it.
Induction saves a diagnostic report and raises `ComponentCoverageError` if any
target remains uncovered; no incomplete constitution is published. This gate
remains enabled with atomicity disabled. The final reconstruction report
evaluates the complete restored list. This is a deterministic CAISO runtime
extension, distinct from the original DICAI procedure and from semantic atomicity
checking. It introduces no additional model-call stage and leaves silhouette
clustering unchanged. See [CONSTITUTION_COVERAGE.md](CONSTITUTION_COVERAGE.md).

## CAISO one-quality semantic check

Three complete CAISO-authored requests in
`caiso/prompts/method/constitution_atomicity/` enforce the existing
one-quality rule at runtime. Each changed chairperson list is reviewed for
independently satisfiable requirements. Compound items receive bounded split
revisions; unflagged items are preserved in Python. The repaired list is reviewed
again before reconstruction or publication. A repeated verified list needs no
new review. A failed revision retains the last verified working list; unverified
seed-only lists cannot become settled output.

The existing split and abstraction-repair requests also include the
generalization instruction and retain actual component IDs. Split repair
preserves the quality and logical conditions being compared, allowing
sample-specific names to become semantic roles. These prompt edits add no
entity-removal or component-name check; unflagged atomic items remain unchanged.

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

The later [search protocol v3](OPTIMIZATION_IMPROVEMENTS.md) changes CAISO's
`method/prompt_optimizer/prompt.txt` to request explicit training-error repairs,
with reference targets, downstream outputs and freshly comparable archived
instruction scores. Its complete-request snapshots and the method prompt hash
were updated intentionally. The 29 protected benchmark prompt entries and all
15 source-derived DICAI requests remain unchanged. Bootstrap label ties now
abstain; the earlier descriptions of unchanged tie policy refer to the source
and atomicity migrations, not protocol v3.

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
