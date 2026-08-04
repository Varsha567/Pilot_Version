# Project Plan: Scalable, Confidence-Aware, Cache-Augmented Query Routing for Energy-Efficient LLM Serving

*A defensible plan document — prepared for guide review.*

---

## 1. Base Paper

**PEARL: Performance and Energy Aware Routing for LLMs**
Kouider Chadli, Goetz Botterweck, Takfarinas Saber — *Future Generation Computer Systems*, 176 (2026) 108218, Elsevier. DOI: 10.1016/j.future.2025.108218

This is a peer-reviewed, journal-published (Elsevier, Scopus-indexed) paper — satisfies the "base paper must be a journal" requirement.

---

## 2. What PEARL Does (one paragraph, for context)

PEARL routes each incoming query to the best LLM out of a pool of 4 (Gemma-2B, Llama3.2-3B, Mathstral-7B, Mistral-7B) by predicting **(a)** expected accuracy via a clustering-based Performance Estimator and **(b)** expected energy via per-model regression models (Energy Consumption Meta-Models, EMMs) — both driven by SBERT query embeddings. It then solves a **Mixed Integer Linear Program (MILP)** to assign each query to the model that maximizes accuracy subject to a configurable energy budget. It reports 58.17%/74.66% accuracy on MMLU/ARC-C at 0.77/0.95 Wh per query, beating baselines like MetaLLM and FORC on the accuracy-energy Pareto front (Hypervolume 0.245/0.371).

---

## 3. Our Problem Statement (final)

> PEARL's own paper identifies three unresolved limitations: **(1)** its MILP-based routing solver does not scale — latency grows from 0.20s (8 queries) to 25.5s (2048 queries), making it unusable for real-time/high-throughput deployment; **(2)** its Performance Estimator provides no uncertainty/confidence signal on its predictions, which the authors flag under "Reliability considerations" as a reliability gap; **(3)** every query is routed independently, with no mechanism to exploit repeated or similar traffic, a realistic characteristic of production LLM deployments.
>
> We propose an extended routing system that keeps PEARL's dual-predictor framework (Performance Estimator + Energy Consumption Meta-Model) unchanged, and adds three components to address these three gaps: **(1)** an NSGA-II-based multi-objective assignment algorithm replacing the MILP solver, **(2)** a confidence score layered on top of the cluster-based Performance Estimator, and **(3)** a semantic routing cache that reuses prior routing decisions for similar queries. We evaluate whether this system retains PEARL's accuracy–energy Pareto quality (Hypervolume, Pareto Front Size) at substantially lower routing latency, on the same MMLU and ARC-C benchmarks with the same four LLMs.

---

## 4. System Name & Architecture

We call the system **[Team to decide a name — e.g. "PEARL-X" / "FastPEARL" / "C²R"]**. One name used consistently everywhere (title, diagrams, code, tables) — this matters for how coherent the paper looks.

```
Incoming Query
      ↓
 SBERT Embedding (φ) — same embedding model as PEARL (all-MiniLM-L6-v2)
      ↓
 Semantic Cache Lookup (cosine similarity ≥ τ against cached query embeddings)
      ↓
   HIT? ──────────────────No───────────────────┐
      │ Yes                                     ↓
      ↓                              Performance Estimator (clustering, from PEARL)
 Validate cached decision:                      +
  - similarity ≥ τ?                    Confidence Score (NEW — variance/
  - energy budget E unchanged?          distance-based uncertainty on cluster prediction)
  - not expired (TTL)?                           +
      ↓                              Energy Meta-Model (EMM, from PEARL, unchanged)
 Reuse (model, acc, energy,                      ↓
  confidence)                        NSGA-II Assignment (NEW — replaces MILP)
      └─────────────────┬────────────────────────┘
                         ↓
                  Call Selected LLM
                         ↓
                      Response
                         ↓
         Update cache (embedding → routing decision, if cache-miss path)
```

---

## 5. Explicit Scope — In / Out (say this to your guide up front)

**Reused from PEARL, unmodified (we do NOT touch these):**
- SBERT embedding model (`all-MiniLM-L6-v2`)
- Clustering-based Performance Estimator methodology (k-means on embeddings, cluster-average accuracy)
- Energy Consumption Meta-Model methodology (regression per LLM: KNN/XGBoost/MLP)
- The 4-LLM pool: Gemma-2B, Llama3.2-3B, Mathstral-7B, Mistral-7B
- MMLU + ARC-C datasets, same train/val/test splits, same Exact Match accuracy metric
- One-shot Chain-of-Thought prompting template

**New contributions (what we are actually building):**
1. **NSGA-II routing algorithm** — replaces PEARL's MILP solver for the assignment step.
2. **Confidence-aware clustering** — adds an uncertainty score to the Performance Estimator's accuracy prediction (e.g., using intra-cluster variance or a distance-weighted confidence measure), used to decide when a prediction is reliable enough to trust vs. when to hedge (route conservatively / fall back to a stronger model).
3. **Semantic routing cache** — caches *routing decisions* (not LLM answers), reused for embedding-similar future queries, evaluated under simulated realistic traffic repetition.

**Explicitly out of scope (state this so no one accuses you of scope creep or missing something):**
- We do not retrain or fine-tune any of the 4 LLMs.
- We do not modify the embedding model.
- We do not add new objectives (e.g., latency-SLO, carbon) beyond PEARL's existing accuracy/energy objectives — noted as future work.
- We do not add new LLMs beyond PEARL's original 4.

---

## 6. Why Each Component Is Feasible (no GPU-heavy training required)

| Component | Tooling | Compute needed | Training involved? |
|---|---|---|---|
| NSGA-II | `pymoo` (Python library) | CPU only, seconds–minutes | No training — it's a search/optimization algorithm run at inference time |
| Confidence score | NumPy / basic statistics on existing cluster outputs | CPU only, negligible | No training — a formula applied to PEARL's existing cluster data |
| Semantic cache | Cosine similarity + a dictionary/vector store (or `faiss` for scale) | CPU only, negligible | No training — a lookup structure |
| Performance Estimator (reused) | `scikit-learn` k-means | CPU only | Already lightweight in original paper — no GPU |
| EMM (reused) | `scikit-learn`/`xgboost` | CPU only | Already lightweight in original paper — no GPU (confirmed: their own Table 3 reports training energy of ~0.02–0.18 Wh, i.e., seconds on CPU) |
| LLM inference (generating the raw query→accuracy/energy dataset) | HuggingFace Transformers, 4 models ≤7B | **Requires GPU** — only step with real hardware demand | No training — inference only, not fine-tuning |

**Only one step needs a GPU: running the 4 LLMs on MMLU/ARC-C to produce accuracy+energy measurements per query.** Everything else — including every one of our three novel contributions — runs on a normal laptop CPU.

---

## 7. Hardware & Execution Plan

- **Local laptop** (i5-13420H, 16GB RAM, integrated GPU): used for all novel-component development — NSGA-II, confidence scoring, caching, statistical analysis, plotting, writing.
- **Google Colab (free tier, T4 GPU)**: used *only* for the one-time (or occasional) step of running the 4 LLMs over MMLU/ARC-C to generate the (query, model, accuracy, energy) dataset. Output saved as CSV/JSON and downloaded — this becomes the fixed dataset all subsequent local work operates on.
- **Energy measurement tool**: **CodeCarbon** — the same library confirmed to be used in PEARL's own repo (verified directly from their `emissions.csv` file), run inside the Colab session during LLM inference.
- **Important documented caveat**: PEARL's paper states energy was measured on an NVIDIA RTX A100; their repo's actual emissions log shows an RTX A500 Laptop GPU instead. Our own measurements will be on Colab's T4. We will **clearly report which hardware each number came from** and focus our claims on *relative* comparisons (% latency reduced, % Pareto-quality retained) rather than claiming exact absolute-Wh equivalence with PEARL's numbers — this is standard, defensible practice when hardware differs, and we state it explicitly rather than hiding it.

---

## 8. Baselines for Comparison

**Reused directly from PEARL's own Table 4/5 (cited, not re-implemented):** Oracle, Random, Individual LLMs, Clustering+TF-IDF/Embedding/RoBERTa, MetaLLM, FORC, PEARL (MILP).

**Not re-implemented (explicitly justified):** RouteLLM (binary-only, cost-not-energy-based — structurally different problem framing) is discussed qualitatively in Related Work, not benchmarked numerically, to keep scope realistic. This will be stated as a deliberate, explained decision, not an oversight.

---

## 9. Results Tables We Will Produce

**Table A — Main accuracy/energy comparison (same format as PEARL Table 4)**

| Method | MMLU Acc% (Energy Wh) | ARC-C Acc% (Energy Wh) |
|---|---|---|
| PEARL (MILP) [reported] | 58.17 (0.77) | 74.66 (0.95) |
| Ours: NSGA-II only | — | — |
| Ours: NSGA-II + Confidence | — | — |
| Ours: Full system (+ Cache) | — | — |

**Table B — Pareto front quality (same format as PEARL Table 5)**

| Method | HV (MMLU/ARC-C) | PFS (MMLU/ARC-C) |
|---|---|---|
| PEARL [reported] | 0.245 / 0.371 | 16 / 18 |
| Ours | — | — |

**Table C — Scalability (our headline result, extends PEARL Fig. 4)**

| Batch size | PEARL MILP latency | Ours (NSGA-II) latency | Speedup |
|---|---|---|---|
| 512 | >6s | — | — |
| 2048 | 25.5s | — | — |

**Table D — Cache performance (under simulated repetition, since MMLU/ARC-C have no natural duplicates)**

| Injected duplicate rate | Hit rate | Latency saved | Accuracy delta vs. no-cache |
|---|---|---|---|
| 10% / 30% / 50% | — | — | — |

**Table E — Confidence calibration (does the confidence score mean anything?)**

| Confidence bucket | Predicted accuracy | Actual accuracy | Calibration gap |
|---|---|---|---|

This ablation structure (A → B → C → D → E) is exactly what a reviewer/guide wants: each component's contribution is isolated and independently justified, not just one big combined number.

---

## 10. Success Criteria (define this NOW, in writing, before running any experiments)

Stated explicitly so a guide can't later claim you moved the goalposts:

> Our system is considered successful if it retains **≥90% of PEARL's Hypervolume** (Pareto quality) while achieving **at least a 5–10× reduction in routing latency** at large batch sizes (≥512 queries), and if the confidence score shows measurable calibration (higher-confidence buckets have higher actual accuracy) and the cache shows a positive hit rate with no meaningful accuracy degradation on cache hits, under realistic simulated repetition rates.

If actual results differ, that's still a reportable, honest finding (a documented latency-vs-quality tradeoff curve is itself a valid contribution) — not a failed project.

---

## 11. Timeline (4–6 months)

| Phase | Duration | Deliverable |
|---|---|---|
| 1. Setup & reproduction | Weeks 1–3 | Reproduce PEARL's baseline pipeline (embedding, clustering, EMM) on our data; verify we can get numbers close to their Table 2 |
| 2. LLM inference dataset generation | Weeks 3–5 | Run 4 LLMs on MMLU+ARC-C on Colab, log accuracy+energy via CodeCarbon |
| 3. NSGA-II implementation | Weeks 5–8 | Replace MILP with NSGA-II, validate against PEARL's Pareto front |
| 4. Confidence scoring | Weeks 8–10 | Add uncertainty layer to Performance Estimator, calibration analysis |
| 5. Semantic caching | Weeks 10–12 | Implement cache, run simulated-repetition experiments |
| 6. Full integration + ablations | Weeks 12–16 | Combine all 3 components, produce Tables A–E |
| 7. Writing & submission prep | Weeks 16–20 | Paper draft, IEEE/conference formatting, review |
| Buffer | Weeks 20–24 | Address guide feedback, revisions, resubmission if needed |

---

## 12. Anticipated Guide Questions — Prepared Answers

**Q: "Why didn't you just re-implement PEARL from scratch instead of relying on their reported numbers?"**
A: We are reproducing their pipeline locally (Phase 1) to validate we understand and can approximately replicate their methodology, but we cite their *published, peer-reviewed* numbers as the baseline for the MILP comparison specifically, since exact hardware reproduction (A100) isn't available to us — this is standard practice when extending a published system rather than fully re-deriving it, and we disclose this explicitly.

**Q: "Isn't NSGA-II just swapping one solver for another — where's the novelty?"**
A: The novelty is in *quantifying the tradeoff* using PEARL's own evaluation metrics (Hypervolume, Pareto Front Size, latency-vs-batch-size) — this is literally the scalability gap PEARL's authors name as future work in their Conclusion.

**Q: "How do you know your confidence score is actually meaningful, not just noise?"**
A: We validate it with a calibration table (Table E) — bucketing predictions by confidence level and checking whether higher-confidence buckets really do have higher actual accuracy. If they don't, that's a reportable negative result, not something we hide.

**Q: "MMLU/ARC-C don't have duplicate queries — isn't your caching experiment fake?"**
A: We explicitly simulate realistic production repetition rates (10/30/50%) rather than claiming native benchmark gains — this is the standard evaluation methodology used in the semantic-caching literature (e.g., GPTCache-style evaluations), and we state this limitation transparently in our methodology section.

**Q: "Why these three components together — aren't they unrelated?"**
A: All three attack the same root problem PEARL's own paper identifies — MILP-based routing does not scale to real-time/high-throughput serving. NSGA-II fixes the per-batch solve time; caching reduces how many queries ever reach the solver; confidence scoring lets the system make faster, cheaper decisions when uncertainty is low. It's one coherent scalability story, not three disconnected ideas.

**Q: "What if your results are worse than PEARL's?"**
A: Our success criterion (Section 10) is defined as a latency-vs-quality tradeoff (≥90% Hypervolume retained at 5–10× latency reduction), not "must beat PEARL on accuracy." A documented tradeoff, honestly reported, is a valid systems-paper contribution — this is how the field evaluates approximate algorithms against optimal-but-slow ones.

**Q: "Did you verify the GitHub code actually matches the paper?"**
A: Yes — we reviewed `optimal_v1` (Chadlikouider/optimal_v1) directly. We found it lists a different model pool than the paper (Llama-3-8B/Yi-1.5-9B/Gemma-7B/Mistral-7B vs. paper's Mistral-7B/Mathstral-7B/Llama3.2-3B/Gemma-2B), and its `emissions.csv` energy logs show an RTX A500 Laptop GPU rather than the A100 stated in the paper's Section 4.3.2. We have emailed the corresponding author for clarification and are proceeding by implementing the methodology directly from the paper's equations (Sections 3.2–3.4) rather than assuming the repo is a verbatim match.

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| NSGA-II hyperparameters (population size, generations) affect result quality significantly | Run a small hyperparameter sweep, report sensitivity, pick best-performing stable config |
| Colab session limits interrupt LLM inference runs | Checkpoint/save partial results frequently; process in smaller batches |
| Energy numbers not hardware-comparable to PEARL's | Report clearly labeled, separate hardware notes; emphasize relative metrics |
| Corresponding author doesn't respond in time | Proceed independently from the paper's equations (already fully specified in Sections 3.1–3.4) — repo access is a nice-to-have, not a blocker |
| Confidence score shows poor calibration | Report as an honest negative/mixed result with analysis of why — still a valid, defensible finding |

---

*This document should be treated as a living plan — update Section 9 tables with real numbers as experiments complete, and revisit Section 10's success criteria only if there's a well-justified reason, documented explicitly.*
