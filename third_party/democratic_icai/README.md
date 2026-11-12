# Original DICAI source and trajectory adaptation

The user supplied `Democratic-ICAI-C8D5` on 2026-11-08. Its eight files are
preserved byte for byte in `original/`, under their original MIT license.
`PROMPTS.json` records source hashes, all 15 complete request templates and the
explicit trajectory adaptations. The original files are reference material;
CAISO does not import or execute their Azure, AutoGen or local-model entry points.

`prompts/` contains complete original requests extracted from the supplied code.
`caiso/prompts/dicai/` contains their complete editable trajectory versions.
Original creative-writing persona constants are retained under `prompts/personas/`;
CAISO's configured dataset personas remain the debate participants.

The permitted adaptations are response/story terminology to trajectory
terminology, human preference wording to provided preference labels, structured
CAISO execution records as evidence, and the documented reasoning, debate and
abstraction guidance for component criteria and reusable principles, plus
training-only task-score feedback for reasoning and all debate variants. The
score-feedback adaptation exposes absolute A/B scores, their signed gap and
score relation separately from trajectory evidence; blind reconstruction remains
score-free. Post-inference regret drives optional minimization on fully scored
data. Existing A/B labels and exact-tie participation are retained. See
`docs/PREFERENCE_COLLECTION.md` for the complete behavior. Each principle
starts with `Select the trajectory`; no particular following connector is required.

The active source is the supplied code, which includes explicit sampled
self-consistency, iterative refinement, one-quality rules and reconstruction
gates. The paper source in `paper/` is retained as a historical reference, under
its separate CC BY-NC-SA 4.0 license. It is no longer the active prompt source.
