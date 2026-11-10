# Component coverage in constitutions

The DICAI trajectory prompts ask reasoners and debaters to inspect each component
for differences that help explain the preference. Chairpersons retain supported
criteria for different components, and abstraction preserves exact component
IDs. Each principle starts with `Select the trajectory where`, immediately
followed by its actual component ID as the subject and one observable criterion.
Every system constitution must contain at least one principle for each workflow
component; every block constitution must cover each selected block member. Each
principle evaluates one quality and expresses a reusable rule without
sample-specific entity names.
The original DICAI request formats, reasoning strategies and three-person debate
remain in use. Original reference files are preserved under
`third_party/democratic_icai/original/`.

## Coverage after clustering

CAISO collects the active workflow IDs for system induction and the selected
block members for block induction. Standalone DICAI calls can supply
`component_ids`; otherwise IDs come from trajectory execution records, or the
`block:...` scope. Text-only preference data without execution IDs receives a
`not_applicable` report.

After the existing silhouette clustering, abstraction, atomicity handling and
optional minimization, CAISO checks which target IDs are the subject immediately
after `Select the trajectory where` and were observed in retained preference
trajectories. Mentioning a component later in a criterion does not cover it.
References such as "final answer" do not stand in for `answer_generator`.
Partial IDs such as `retriever` inside `wikipedia_retriever` do not count.

For each missing component, CAISO restores an existing settled debate principle
that evaluates it as the subject. All target components named by that rule must
have executed in its source preference pair. Rules supported by more distinct pairs are preferred;
ties use fewer words and then source order. Repeated copies within one pair do
not increase its distinct-pair support. An interaction rule may mention multiple
components while expressing one quality, but covers only its subject. The other
components need their own principles.

Restoration copies the settled rule verbatim, retaining its source pair IDs and
settled-principle indices. It introduces no new model call, template-derived
principle, alias substitution or entity-name rewrite. Its atomicity has the
same verification as its settled source; disabling atomicity also disables that
source's semantic check. Principles using another connector, generic aliases,
placeholder IDs or unknown subjects are rejected by format/attribution validation. Reasoning must supply
actual component criteria; dimensions alone no longer generate fallback rules.
The final reconstruction report evaluates the resulting complete list.

If no eligible settled rule exists, induction raises `ComponentCoverageError`
and does not publish a constitution. Its coverage report is saved before the
error, identifying missing and unobserved components. More informative training
preference comparisons or better debate outputs are needed to cover them.
This coverage requirement remains enabled when semantic atomicity is disabled.
No requirement forces every pair to cover every component. This check measures
explicit attribution and source provenance; it does not prove the rule is semantically correct or that its component caused the
preference.

Clustering and the selected k are unchanged. Restoration can make the final
principle count exceed k. `cluster_ids`, `abstraction_outputs`, `clusters` and
`retained_cluster_ids` keep their existing meanings; restored rules are tracked
separately in the coverage report rather than assigned invented cluster IDs.
Optional minimization may mark a rule dropped before coverage restores it;
the final `principles` list and coverage report are authoritative.

## Saved reports

Each constitution includes `metadata.component_coverage`. A separate
`component_coverage.json` is also saved:

- Beside each provenance-specific cached constitution.
- At `.internal/dicai/system/component_coverage.json` for the latest
  system induction attempt, including a failed coverage check.
- In iteration `00_system_constitution` and `04_block_constitution` folders when
  those stages save their constitution.

The report records `expected_components`, `observed_components`,
`covered_before`, `covered_after`, `restored`, `missing_components`,
`insufficient_evidence_components` and `unobserved_components`. It provides
one-based `principle_indices_by_component` and
`unattributed_principle_indices` for reviewing the final list. Its status is
`complete`, `insufficient_evidence` or `not_applicable`. An
`insufficient_evidence` report is diagnostic output from a failed induction,
not an accepted constitution.

This is a CAISO extension to the source DICAI procedure, identified by protocol
`democratic-icai-c8d5-source-trajectory-v6-required-component-principles`. Prompt,
implementation and coverage-target fingerprints separate induction caches.
Successful requests still resume within a matching induction. Start a new
optimization run to regenerate older saved constitutions; an existing run's
checkpoint retains its constitution.
