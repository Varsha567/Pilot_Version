# Realistic Day-by-Day Project Plan (Starting Aug 10, 2026)

*Covers Review-1 prep, full implementation, and results — with best practices baked in.*

---

## Ground Rules / Best Practices (apply throughout, not just one phase)

1. **Version control from Day 1.** Create a GitHub repo (private is fine) the moment you start. Commit at the end of every working session, even small ones — this is your safety net if something breaks, and it's evidence of consistent progress if a guide asks.
2. **Save incrementally, never in one big batch.** Every long-running script (especially Colab LLM inference) should write results to CSV *as it goes*, not only at the end — protects you from session timeouts/crashes losing hours of work.
3. **Keep a daily log.** A single running text file: date, what you did, what broke, what you decided. Takes 2 minutes a day, saves you when your guide asks "why did you choose X" three weeks later and you've forgotten your own reasoning.
4. **Test on a tiny sample before scaling up.** Before running 600 queries × 4 models, run 5 queries × 1 model first, check the pipeline works end-to-end, *then* scale. This single habit prevents the most common failure mode (discovering a bug after 2 hours of compute time).
5. **Weekly sync with your guide**, even a short 10-minute one, not just at formal Reviews. Reduces the "guide pinpoints everything at Review-2" risk, since they've already seen incremental progress.
6. **Every deliverable gets a rationale sentence ready**, per the Q&A prep already in your plan doc — review it before each guide interaction, not just once.

---

## Phase 0 — Review-1 Preparation (Aug 10–13)

**Goal: a polished, defensible 10-slide deck (structure already agreed), no results yet — that's fine and expected at Review-1.**

| Date | Task |
|---|---|
| **Aug 10 (Mon)** | Finalize title with mam's chosen option. Build Slides 1–4 (Title, Problem, Base Paper, Gaps-in-their-own-words). |
| **Aug 11 (Tue)** | Build Slides 5–7 (Architecture diagram, Scope table, Feasibility). Draft speaker notes for the 90-second pitch. |
| **Aug 12 (Wed)** | Build Slides 8–10 (Evaluation plan templates, Success criteria, Novelty-vs-prior-work). Full run-through as a team, time it. |
| **Aug 13 (Thu)** | Buffer day: incorporate any last feedback from teammates/guide pre-check, polish visuals, rehearse Q&A prep once more. |
| **Review-1 (whenever scheduled)** | Present. Afterward: log every question asked and how you answered — this becomes your Review-2 prep sheet. |

---

## Phase 1 — Setup & Environment (Aug 14–16)

| Date | Task |
|---|---|
| **Aug 14 (Fri)** | Create GitHub repo + folder structure (`/data`, `/notebooks`, `/src`, `/results`, `/docs`). Set up Hugging Face account + token. Request Llama3.2-3B gated access **today** (approval can take time — don't block later phases on it). |
| **Aug 15 (Sat)** | Install & test locally: `scikit-learn`, `pymoo`, `xgboost`, `sentence-transformers`, `codecarbon`, `pulp`. Run the 5-minute CodeCarbon sanity test (from earlier) — confirm it prints a real Wh number. |
| **Aug 16 (Sun)** | Set up Colab notebook: install `transformers`, `bitsandbytes`, `accelerate`, `codecarbon`. Load **one** model (start with Gemma-2B, smallest) in 4-bit quantization, run **one** test query end-to-end with energy logging. Confirm the full mini-pipeline works before scaling. |

---

## Phase 2 — LLM Dataset Generation (Aug 17–21)

| Date | Task |
|---|---|
| **Aug 17 (Mon)** | Finalize query subsample from MMLU + ARC-C (confirm size with guide — 300–600 per dataset recommended given free-tier limits, more if guide wants full test sets). Prepare PEARL's exact CoT prompt template + hyperparameters in code. |
| **Aug 18 (Tue)** | Run Gemma-2B + Llama3.2-3B (the two smaller models) across the full query set on Colab, logging accuracy (Exact Match) + energy (CodeCarbon) incrementally to CSV. |
| **Aug 19 (Wed)** | Run Mathstral-7B + Mistral-7B (the two larger, quantized models) across the full query set. Expect this session to take longer — block 3+ hours. |
| **Aug 20 (Thu)** | Buffer/catch-up day for any Colab session interruptions, reruns, or missing rows. Consolidate into one clean `dataset.csv`. |
| **Aug 21 (Fri)** | Sanity check: compare your accuracy/energy numbers' *ordering* against PEARL's Table 2 (does Gemma stay cheapest, does the 7B models stay highest/most variable in energy?). Document findings in your daily log — this becomes a citable methodology-validation paragraph later. |

---

## Phase 3 — Reproduce PEARL's Predictors + MILP Baseline (Aug 22–27)

| Date | Task |
|---|---|
| **Aug 22 (Sat)** | Generate SBERT embeddings for all queries. Run k-means clustering, tune K empirically (try a few values, pick based on cluster coherence). |
| **Aug 23 (Sun)** | Compute per-cluster average accuracy (Performance Estimator) **and** per-cluster standard deviation (your confidence signal — do this now, it's the same step). |
| **Aug 24 (Mon)** | Train EMM regressors (XGBoost, one per LLM) on query embedding → energy. Evaluate with MSE/MAPE, same as PEARL's Table 3 format. |
| **Aug 25 (Tue)** | Implement the MILP formulation (Section 3.3 of PEARL) using PuLP — decision variables, objectives, constraints, exactly as specified. |
| **Aug 26 (Wed)** | Run MILP baseline at multiple batch sizes (8, 32, 128, 512, 2048 — matching PEARL's Fig. 4 test points), logging latency + energy overhead via CodeCarbon on your own hardware. |
| **Aug 27 (Thu)** | Buffer/debugging day. Confirm MILP output accuracy/energy roughly tracks PEARL's Table 4 trend (even if absolute numbers differ due to hardware/sample size). |

---

## Phase 4 — NSGA-II Implementation (Aug 28 – Sep 2)

| Date | Task |
|---|---|
| **Aug 28 (Fri)** | Implement 2-objective NSGA-II (max accuracy, min energy) using `pymoo`, using the *same* predicted accuracy/energy inputs the MILP baseline used. |
| **Aug 29 (Sat)** | Tune NSGA-II hyperparameters (population size, number of generations) — run a small sweep, document tradeoffs (solution quality vs. speed). |
| **Aug 30 (Sun)** | Run NSGA-II at the same batch sizes as the MILP baseline (Aug 26). Log latency + energy overhead. |
| **Aug 31 (Mon)** | Build the direct comparison: NSGA-II vs. your own MILP, same hardware, same data — Table C (latency) draft. |
| **Sep 1 (Tue)** | Compute Hypervolume + Pareto Front Size for both MILP and NSGA-II fronts — Table B draft. |
| **Sep 2 (Wed)** | Buffer/debugging day. |

---

## Phase 5 — Confidence Integration (Sep 3–5)

| Date | Task |
|---|---|
| **Sep 3 (Thu)** | Decide and implement the confidence-usage rule (e.g., low-confidence queries get routed toward a stronger/safer model, or their predicted accuracy gets down-weighted in the objective). |
| **Sep 4 (Fri)** | Run calibration analysis: bucket predictions by confidence level, compare predicted vs. actual accuracy per bucket — Table E draft. |
| **Sep 5 (Sat)** | Buffer/debugging + write the confidence methodology paragraph while it's fresh (include the CP-Router/confidence-token differentiation from Related Work). |

---

## Phase 6 — Semantic Cache (Sep 6–8)

| Date | Task |
|---|---|
| **Sep 6 (Sun)** | Implement cosine-similarity cache on query embeddings (cache the routing decision, not the LLM's answer — as scoped earlier). |
| **Sep 7 (Mon)** | Build synthetic duplicate-injection test harness (10/30/50% repetition rates), run the sweep, log hit rate + latency saved — Table D draft. |
| **Sep 8 (Tue)** | Buffer/debugging day. |

---

## Phase 7 — Full Integration & Ablation (Sep 9–12)

| Date | Task |
|---|---|
| **Sep 9 (Wed)** | Wire all three components into one pipeline (cache → predictors+confidence → NSGA-II). |
| **Sep 10 (Thu)** | Run the full 4-row ablation (MILP baseline → NSGA-II only → +Confidence → +Cache) across your dataset — Table A + Table B final versions. |
| **Sep 11 (Fri)** | Generate all plots: Pareto front comparison (MILP vs. NSGA-II), latency-vs-batch-size curve, calibration curve, cache hit-rate curve. |
| **Sep 12 (Sat)** | Buffer/debugging + sanity-check every table against your Section 10 success criteria (≥90% Hypervolume retained, 5–10× latency reduction). |

---

## Phase 8 — Writing & Review-2 Prep (Sep 13–18)

| Date | Task |
|---|---|
| **Sep 13 (Sun)** | Draft Methodology section (can mostly reuse your plan doc's language, now with real specifics filled in). |
| **Sep 14 (Mon)** | Draft Results section around the finalized tables/plots. |
| **Sep 15 (Tue)** | Draft/update Related Work section (PEARL + CP-Router/confidence-token differentiation + MetaLLM/FORC citations). |
| **Sep 16 (Wed)** | Update Review-1 deck into Review-2 deck: swap placeholder tables for real results, add a "What We Learned / Limitations" slide. |
| **Sep 17 (Thu)** | Full team rehearsal, anticipate questions from Review-1 log + new results-specific questions ("why is X lower than Y," etc.). |
| **Sep 18 (Fri)** | Buffer day / final polish. |

---

## Summary Timeline at a Glance

| Phase | Dates | Duration |
|---|---|---|
| 0. Review-1 prep | Aug 10–13 | 4 days |
| 1. Setup | Aug 14–16 | 3 days |
| 2. LLM dataset generation | Aug 17–21 | 5 days |
| 3. Predictors + MILP baseline | Aug 22–27 | 6 days |
| 4. NSGA-II | Aug 28–Sep 2 | 6 days |
| 5. Confidence | Sep 3–5 | 3 days |
| 6. Cache | Sep 6–8 | 3 days |
| 7. Integration + ablation | Sep 9–12 | 4 days |
| 8. Writing + Review-2 prep | Sep 13–18 | 6 days |
| **Total** | **Aug 10 – Sep 18** | **~5.5 weeks** |

This includes a buffer day built into nearly every phase — if everything goes smoothly, you finish early with time to deepen results (e.g., scale up the dataset further, add the RouteLLM supplementary comparison). If something breaks, you have room to absorb it without the whole timeline collapsing.

---

## Division of Labor (adjust to your actual team size/skills)

A reasonable 3–4 person split, so phases can run partially in parallel rather than strictly sequential:

- **Person A (ML/data):** Phases 2, 3 (predictors + EMM) — owns the dataset and prediction pipeline.
- **Person B (optimization):** Phase 4 (NSGA-II) + Phase 3's MILP baseline — owns the routing algorithms.
- **Person C (systems/eval):** Phase 6 (cache) + Phase 7 (integration, ablation, plots) — owns bringing it all together.
- **Person D (or shared):** Phase 5 (confidence) + writing throughout (Phase 8) + slide decks (Phase 0) — owns documentation and communication, since this needs constant updating as others' work lands.

Everyone attends the weekly guide sync and contributes to the daily log for their own component.
