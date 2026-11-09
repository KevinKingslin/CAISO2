# CAISO proposal overview

Original vector artwork based on `CAISO_proposal.pdf` supplied by the user
(Sections 4.1-4.7 and the block-optimization steps of Algorithm 1).
This is a diagram of the proposed method,
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
- After a rejected candidate, the accepted state is retained. Attribution and
  partitioning are recomputed on subsequent iterations (Algorithm 1).

The figure emphasizes candidate generation: alternative prompts from prompt
search, weights trained using constitution-labeled preferences and DPO, and
sampled model or parameter updates. Component proposals are combined within
the selected block and ranked or filtered jointly using its constitution.
Workflow adaptation is omitted at the user's request. The figure focuses on
the current workflow's component configurations: block discovery, constitution
induction, and candidate generation. The block validation gate remains visible.
The initial pair of configurations used to induce the local constitution is
shown separately from the subsequent constitution-guided optimization proposals.
The proposal does not prescribe a specific algorithm for generating that initial
pair; the figure does not add one. Detailed sampling budgets and batch aggregation
belong in the method text.

## Rebuild

Requires Python 3, ReportLab, Poppler, and Arial fonts. The script uses macOS
font paths; change the two `TTFont` registrations on another platform.

```sh
python3 figures/caiso/draw_overview.py
pdftoppm -singlefile -r 300 -png output/pdf/caiso_overview.pdf output/pdf/caiso_overview
```
