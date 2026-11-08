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
CAISO execution records as evidence, and the two previously agreed sentences
about component criteria and preservation of component references. Each principle
starts with `Select the trajectory`; no particular following connector is required.

The active source is the supplied code, which includes explicit sampled
self-consistency, iterative refinement, one-quality rules and reconstruction
gates. The paper source in `paper/` is retained as a historical reference, under
its separate CC BY-NC-SA 4.0 license. It is no longer the active prompt source.
