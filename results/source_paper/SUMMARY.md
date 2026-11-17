# Run summary (provisional until the Meditron retry is applied)

Written 2026-11-15. All numbers are in `results/aggregate/`; row-level generations and scored rows stay local (`results/raw`, `results/scored`).

## What ran
- Items: 600 (200 MedSafetyBench, 200 MMedBench English, 200 WMDP-bio), seed 13, 27 conditions (baseline, 5 dialects, 3 translations, 9 English-base and 9 native-base code-switch).
- Models: Gemma-3-27B, MedGemma-27B, Mistral-Small-3.2-24B, Llama-3.1-8B, Llama3-Med42-8B, Meditron-7B. Generation complete for both experiments: 87,216 Experiment 1 rows and 27,816 Experiment 2 rows (Experiment 2 uses MMedBench only), plus the gate-failed sensitivity set (7,392 and 3,102 rows).
- Judge: Qwen3-32B (Experiment 1). Classifiers: Llama Guard 4, WildGuard, PolyGuard, Qwen3Guard (input and output, all rows).
- Rendering status of the 16,200 condition rows: ok 14,528; failed the judge-panel gate 1,232; error 248; failed checks 125; needs review 59 (64 low-similarity items excluded, option A); ok after triage 8. Seven code-switch conditions exceed the 15% unrendered flag (sanity check 5).

## Sanity checks (`sanity_checks.json`)
Pass: 1 to 4, 7 to 9, 11, 12. Warn: 5 (seven conditions above 15% unrendered, reported), 10 (3 of 20 temperature-0 re-runs differ, batched GPU server). Check 6 (unparseable above 20%): fails for Meditron-7B only, in some Experiment 1 cells, after the lenient answer rule; a repair of its 558 empty replies is queued (job 35725208).

## Headline results (descriptive; intervals in the tables)
- Experiment 1: harmful compliance is higher than on SAE under dialects, translation and code-switching for the five instruction-tuned models; the largest paired gaps are +0.16 to +0.37 on native-base code-switching and translation. External classifiers lower residual risk by 4 to 12 points and leave the ordering across conditions unchanged. False refusal is at most 0.01 for the five instruction-tuned models. Jamaican English is within the range of the other dialects (H5 not supported).
- Experiment 2: dialects change accuracy little (paired gaps -0.09 to +0.01); translation costs 0.02 to 0.32; ECE and AUROC worsen under translation; verbalised confidence behaves as a silent failure for the general models.
- Hypotheses H1 to H5 are reported descriptively, as the spec forbids significance tests across the grid.
- Gate-failed renderings: harmful compliance 0.05 lower than confirmed renderings of the same cell (0.01 to 0.10), accuracy not distinguishable; the fully rejected stratum has 36 harmful-request rows.

## Did not go to plan
- Rendering: one re-render round only (author decision); seven conditions above the 15% flag; 64 low-similarity items excluded without human review; no native-speaker validation.
- Meditron-7B: 558 empty multiple-choice replies (judge labelled them refuse), 85% of its verbalised replies unparseable, accuracy at chance. Reported as a base model that does not follow the format.
- Mistral and the first three L40 jobs failed once each (HF loading flags, concurrent writes to one pilot file); fixed, no data lost.

## Post hoc changes (all in `docs/DEVIATIONS.md`)
1. Annotator prompt v1 replaced by the judge-style prompt v2 (annotator qualification).
2. One re-render round instead of three; sensitivity run on judge-rejected renderings added.
3. Verbalised-confidence parser revised once, for every model (2026-11-14); Experiment 2 verbalised replies of four models re-asked and re-parsed.
4. Answer accuracy for multiple-choice items uses a lenient, model-independent letter rule with option-text validation; instruction following stays strict (2026-11-15).
5. Meditron empty replies retried without the stop sequence (2026-11-15), pending.
6. `check_results.py` row-count check fixed to count MMedBench only for Experiment 2 (checker bug, not a result).
7. Added analysis (2026-11-15, after the run, exploratory): confidence as a trust flag, TPR/FNR/FPR/TNR at tau = 0.8 with 0.6 and 0.9 as sensitivity rows (`exp2_confusion_*`, `fig_confusion_data`).
