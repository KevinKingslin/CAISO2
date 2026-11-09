# CAISO proposal overview

Original vector artwork based on `CAISO_proposal.pdf` supplied by the user
(Sections 4.1-4.8 and Algorithm 1). This is a diagram of the proposed method,
not a statement of experimental results or implementation completeness.

## Deliverables

- `../../output/pdf/caiso_overview.pdf`: font-embedded vector artwork, 7.5 inches
  wide. Use this in a full-width, two-column paper figure.
- `../../output/pdf/caiso_overview.svg`: editable text and vector geometry.
- `../../output/pdf/caiso_overview.png`: 300 dpi preview/export.
- `figure.tex`: suggested LaTeX insertion and caption.

The design follows the three rounded blue/orange/green stage panels and thin
arrows in Figure 1 of [Democratic ICAI: Debating Our Way to Steering Principles
from Preferences](https://arxiv.org/abs/2606.28294v2), Kingslin et al. (2026).
The layout and all diagram geometry were drawn afresh; no paper images or
illustrations are embedded. Cite the DICAI paper in the manuscript when
discussing constitution induction.

## Interpretation

- Edge effects use repeated end-to-end executions with the proposal's
  type-specific baseline intervention; the red dashed edge illustrates one
  intervention, not permanent edge deletion.
- Dashed outlines illustrate a non-overlapping Leiden-CPM partition. The toy
  graph and partition are schematic, not measured causal results.
- The block preference label comes from complete trajectories evaluated by
  the task objective and system constitution. Only the selected block's
  configurations differ between the two runs.
- The green edge in the third panel illustrates one added connection. All
  workflow modifications must preserve a DAG and use the fixed component library.
- Workflow search is conditional. If neither K accepted block updates nor a
  plateau triggers search, iteration continues without a workflow proposal.
- After a rejected candidate, the accepted state is retained. After accepted
  structural changes, the system constitution is refreshed. Attribution and
  partitioning are recomputed on subsequent iterations (Algorithm 1).
- Section 4.2 calls the system constitution "fixed," while Section 4.8 and
  Algorithm 1 explicitly refresh it after an accepted structural change. The
  diagram follows the latter, more specific rule. The paper should reconcile
  that wording.

To keep the overview readable, details such as the LLM initializations,
per-optimizer algorithms, sampling budgets, and batch aggregation belong in
the method text. The diagram preserves the global acceptance rules, rather
than suggesting that better local scores alone justify an update.

## Rebuild

Requires Python 3, ReportLab, Poppler, and Arial fonts. The script uses macOS
font paths; change the two `TTFont` registrations on another platform.

```sh
python3 figures/caiso/draw_overview.py
pdftoppm -singlefile -r 300 -png output/pdf/caiso_overview.pdf output/pdf/caiso_overview
```
