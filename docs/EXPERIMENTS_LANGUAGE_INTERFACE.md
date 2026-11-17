# Experiments for the language-as-interface paper

Paper: "Language as the Interface of Power: Co-Constitution, the Expression and Utility Gaps, and AI Safety in the Digital Language Divide". Proposition P1 (safety gradient) is the target. Status of everything below: planned, nothing here has been run for this paper.

## Reuse from the earlier health-domain study
Code, configs and rendering methods were copied from the earlier study (`docs/source_*.md`, `results/source_paper/`). That study already has behavioral results over 27 conditions (SAE, 5 dialects, 3 translation levels, 18 code-switch conditions) and six models, in `results/aggregate/` (`exp1_*` harmful compliance and false refusal, `exp2_*` confidence). They are the behavioral anchor for P1: harmful compliance was higher than on SAE under dialects, translation and code-switching for the five instruction-tuned models (see `results/source_paper/SUMMARY.md`). Caveats carried over: renderings are synthetic, no native-speaker validation, automated judge, seven code-switch conditions above the 15% unrendered flag, health-domain prompts only.

Blinding: that study is under double-blind review. Do not cite its results in the manuscript in a way that identifies authors, and do not make either repository public until review ends.

## E1. Refusal and harmfulness directions under dialect and code-switching (mechanism)
Question: do dialect rewrites and code-switched inputs shift where prompts sit relative to the harmfulness and refusal directions, beyond what standard-language translation does?

Prior work: Arditi et al. 2024 (single refusal direction), Zhao et al. 2025 (harmfulness and refusal are encoded separately), Wang et al. 2025 (English-derived direction acts causally across 14 standard languages; Yoruba weakest), Zhang 2026 (raw cosine uninformative without a control), Oppong et al. 2026 (under 10% of English signal retained in four African languages). None tests dialects or code-switching.

Design:
1. Models: open-weight models already served in this repo (Gemma-3-27B, Llama-3.1-8B, Mistral-Small-3.2-24B; medical variants optional). Record versions.
2. Directions: difference in means on English harmful versus matched benign prompts, at a layer and token position chosen on a held-out English split (record the choice before running other conditions). Separate harmfulness direction (Zhao: harmfulness-labelled prompts) and refusal direction (refused versus complied).
3. Inputs: the same items rendered in SAE, 5 dialects, 3 translations, 9 English-base and 9 native-base code-switch conditions (reuse `scripts/02*` rendering and the quality gates).
4. Measures per condition: projection on each direction; retention of the English harmful-versus-benign separation (`mechanism.retention`); refusal rate from the existing generation pipeline; cosine to the English direction reported beside a random control (`mechanism.random_unit`) and a non-safety control direction.
5. Comprehension check: for each rendered input, back-translation or restatement (existing `scripts/02q_backtranslate.py` pattern), so a fall in projection is not read as a failure to understand.
6. Causal check (secondary): ablate the English direction (`mechanism.ablate`) and steer, then compare refusal change across conditions.
7. Reading: harmfulness projection falls with comprehension intact = recognition gap. Harmfulness holds but refusal falls = recognition-to-refusal gap. Report descriptively with bootstrap intervals over items. No significance tests across the grid; no model ranking.

Limits to state: open-weight models only; synthetic renderings; English-derived directions may be the wrong reference for other varieties; one layer choice; health-domain prompts unless a second source is added.

Not yet built: activation extraction from the models (hooks on residual stream, batching, caching). The arithmetic is in `src/dlsd/mechanism.py` with tests in `tests/test_mechanism.py`. Extraction script is the next piece to write and must be tested on a small model before any large run.

## E2. Behavioral register pilot (cheap, run first)
Hold intent and harm level fixed; vary register only. Reuse the existing generation and scoring pipeline. Add a benign-helpfulness arm (utility gap): answer quality and false refusals by register, not only harmful compliance. Source prompts beyond health (for example a general harmful-request set plus matched benign) so the claim is not domain-bound; choose and record the source before running.

## E3. Feedback-loop simulation (optional, speculative)
Simulate who is counted in preference data by variety, then measure drift of a model fitted on the counted data toward the counted variety. Label as simulation, not evidence about deployed systems.

## Rules (carried over)
- Fixed seeds, resumable stages, run `pytest` before committing.
- Record every model ID, version and decoding setting in the config before running.
- Row-level generations contain responses to harmful requests: local only, gitignored, never quoted.
- Ask the author before changing models, languages, dialects, thresholds or sample sizes.
- No em dashes in any text written for the author or the paper.
- Keep the repository private while the paper and the earlier study are under double-blind review.
