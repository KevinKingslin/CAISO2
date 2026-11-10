# Component coverage in constitutions

The DICAI trajectory prompts ask reasoners and debaters to inspect each component
for differences that help explain the preference. Chairpersons retain supported
criteria for different components, and abstraction preserves exact component
IDs. Each principle still starts with `Select the trajectory`, evaluates one
quality, and expresses a reusable rule without sample-specific entity names.
The original DICAI request formats, reasoning strategies and three-person debate
remain in use. Original reference files are preserved under
`third_party/democratic_icai/original/`.

## Prompt instructions

The debate prompts request principles in the form "Select the trajectory where",
followed immediately by the actual component ID and one reusable criterion. They
ask for at least one separate principle for each component shown in the
trajectories (each identified block member for a block comparison), and preserve
a component's last supported principle when editing or merging. Abstraction
prompts preserve exact evaluated component IDs and their roles. They still
return one principle per cluster, including when a cluster spans components.
These are prompt instructions: parsing, reasoning, clustering, abstraction
outputs, coverage reporting and stopping behavior are unchanged. Prompt compliance
and one separate final principle per component are not runtime guarantees.

The coverage restoration and insufficient-evidence reports described below
predate this prompt revision and are unchanged. Insufficient coverage does not
stop induction.

## Coverage after clustering

CAISO collects the active workflow IDs for system induction and the selected
block members for block induction. Standalone DICAI calls can supply
`component_ids`; otherwise IDs come from trajectory execution records, or the
`block:...` scope. Text-only preference data without execution IDs receives a
`not_applicable` report.

After the existing silhouette clustering, abstraction, atomicity handling and
optional minimization, CAISO checks which target IDs appear explicitly in the
remaining principles and were observed in retained preference trajectories.
References such as "final answer" do not stand in for `answer_generator`.
Partial IDs such as `retriever` inside `wikipedia_retriever` do not count.

For each missing component, CAISO restores an existing settled debate principle
that names it. All target components named by that rule must have executed in
its source preference pair. Rules supported by more distinct pairs are preferred;
ties use fewer words and then source order. Repeated copies within one pair do
not increase its distinct-pair support. An interaction rule may cover multiple
components while still expressing one quality.

Restoration copies the settled rule verbatim, retaining its source pair IDs and
settled-principle indices. It introduces no new model call, template-derived
principle, alias substitution or entity-name rewrite. Its atomicity has the
same verification as its settled source; disabling atomicity also disables that
source's semantic check. Existing generic principles are not silently renamed
or deleted. The final reconstruction report evaluates the resulting list.

If no eligible settled rule exists, the component is reported as insufficient
evidence. More informative training preference comparisons or better debate
outputs are needed to cover it. No requirement forces every pair to cover every
component. This check measures explicit attribution and source provenance; it
does not prove the rule is semantically correct or that its component caused the
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
- At `.internal/dicai/system/component_coverage.json` for the latest successful
  system induction.
- In iteration `00_system_constitution` and `04_block_constitution` folders when
  those stages save their constitution.

The report records `expected_components`, `observed_components`,
`covered_before`, `covered_after`, `restored`, `missing_components`,
`insufficient_evidence_components` and `unobserved_components`. It provides
one-based `principle_indices_by_component` and
`unattributed_principle_indices` for reviewing the final list. Its status is
`complete`, `insufficient_evidence` or `not_applicable`.

This is a CAISO extension to the source DICAI procedure, identified by protocol
`democratic-icai-c8d5-source-trajectory-v5-component-coverage`. Prompt,
implementation and coverage-target fingerprints separate induction caches.
Successful requests still resume within a matching induction. Start a new
optimization run to regenerate older saved constitutions; an existing run's
checkpoint retains its constitution.
