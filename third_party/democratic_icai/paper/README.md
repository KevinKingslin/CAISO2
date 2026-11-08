# DICAI paper prompt source

`appendix_g.tex` is the unmodified Appendix G source from
[arXiv:2606.28294v2](https://arxiv.org/abs/2606.28294v2),
*Democratic ICAI: Debating Our Way to Steering Principles from Preferences*,
Kevin Kingslin, Anish Natekar, Ashutosh Ranjan, Vivek Srivastava, Savita Bhat
and Shirish Karande (2026).

Source archive: https://arxiv.org/src/2606.28294v2

The paper is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
This source and its prompt transcriptions retain that attribution and license;
the anonymous repository's Python code retains its separate MIT license.
Source hashes and the active prompt mapping are recorded in `../SOURCE.json`.

CAISO uses Figures 10–16 for its DICAI instructions, with TeX layout normalized
to plain text and a minimal runtime adaptation for trajectory comparisons.
Debate uses trajectory terminology, the prefix "Select the trajectory where"
and one sentence allowing component/interactions/final output criteria.
Abstraction preserves component references through one added requirement.
Reasoning strategy instructions and Figure 16 judging remain unchanged. The
original source and reference transcription fixture are not edited.
Complete editable requests are under `caiso/prompts/dicai/`.
The benchmark-specific personas and dynamic trajectory evidence are CAISO inputs.
See `docs/PROMPT_AUDIT.md` for the migration and runtime details.
