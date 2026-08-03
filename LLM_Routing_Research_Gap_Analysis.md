# LLM Routing: Literature Landscape, Research Gaps & Undergraduate Project Ideas

*Prepared as a research-scoping document for a 4–6 month undergraduate major project / IEEE-style conference paper.*

> **Honesty note on scope:** A truly exhaustive sweep of NeurIPS/ICML/ICLR/ACL/EMNLP/NAACL/AAAI/IEEE/ACM proceedings plus a full GitHub audit (Step 1–2 as literally specified) is a multi-week systematic-review effort, not something reliable to compress into one chat turn. What follows is a **real, citation-grounded landscape** built from current search (Aug 2026) covering the papers and repos that matter most for scoping a project, organized exactly along your Steps 3–8. I flag where I'm summarizing a field rather than claiming completeness, and I recommend a follow-up systematic pass (Step 1/2, fully expanded) via a background research run — see the note at the end.

---

## Step 1–2 (condensed): Literature & GitHub Snapshot

### A. Foundational / must-know papers

| Paper | Core idea | Routing strategy | Features used | Dataset | Metrics | Strengths | Weaknesses | Remaining gap |
|---|---|---|---|---|---|---|---|---|
| **RouteLLM** (Ong et al., ICLR 2025) | Binary router: strong vs. weak model per query | Learns P(strong wins) from preference data, thresholds on cost budget | Query text embeddings / matrix factorization / BERT classifier | Chatbot Arena (80k battles), MMLU, Nectar | Cost saved at fixed quality (win-rate vs GPT-4) | >85% cost reduction at ~95% GPT-4 quality; strong generalization to unseen model pairs; open-sourced (lm-sys/RouteLLM, ~3.9k★, Apache-2.0) | **Binary only** (2 models); black-box decision, hard to debug; needs recalibration per new model pair; no latency/energy objective | No native support for >2 models, no online adaptation, no multi-objective (cost+latency+energy) formulation |
| **FrugalGPT** (Chen, Zaharia & Zou, 2023, TMLR) | Cascade of LLMs, escalate only when a scorer flags low confidence | Sequential probe-then-escalate with a DistilBERT correctness scorer | Query, prior model's answer | HEADLINES, Overnight, and other cost-benchmark sets | Cost vs. accuracy Pareto curve | Simple, interpretable, works with black-box APIs, no need to retrain LLMs | Cascade order is static; sequential probing wastes compute on rejected calls; single scalar scorer; no latency-SLO awareness | No adaptive cascade **ordering** based on live load/latency, no joint cost+carbon objective |
| **Hybrid LLM** (Ding et al., ICLR 2024) | Route to small vs. large model based on predicted "query difficulty" | Deferral classifier trained on difficulty labels + desired quality level | Query features, difficulty score | Custom quality-labeled sets | Quality-cost tradeoff | Tunable quality knob is user-facing | Only 2-model regime again; a single scalar "difficulty" ignores task-type heterogeneity | No treatment of *multi-dimensional* difficulty (reasoning vs. factual vs. creative) |
| **RouterBench** (Hu et al., 2024) | Standardized benchmark: 405k precomputed outputs, 11 LLMs × 7 tasks (MMLU, MT-Bench, MBPP, HellaSwag, WinoGrande, GSM8K, ARC) | N/A (benchmark, not a router) | — | Own release | Cost-quality AUC | First reusable, precomputed routing benchmark, avoids re-querying APIs (cheap for students!) | Old/small model pool, only 7 fairly easy tasks, no energy/latency ground truth | No coverage of frontier models, no multi-turn/agentic tasks |
| **RouterArena** (Lu et al., ICLR 2026) | Open, automated leaderboard platform for routers across 9 domains, 44 categories, Bloom's-taxonomy difficulty tiers | N/A (benchmark) | — | Own curated set, 8,400 queries | Arena score (log-cost normalized accuracy), cost-ratio, latency, robustness scores | Most comprehensive current benchmark; publishes per-domain breakdowns; live leaderboard | <cite index="19-1">Compares six routing methods (CARROT, RouterDC, GraphRouter, MIRT-BERT, NIRT-BERT, and RouteLLM) across five evaluation dimensions</cite> but each router in the wild uses a **different model pool**, undermining cross-paper comparability; recent follow-up work found judge-scoring artifacts inflate the apparent "routing headroom" by up to 10–24 points on knowledge tasks<cite index="12-1">, with judge scoring deviating from exact-match by up to 10–24 percentage points on knowledge tasks, and truncation affecting up to 65% of responses in certain settings</cite> | Evaluation-artifact robustness itself is an open problem — a genuine, underexplored gap (see below) |
| **LLMRouterBench** (Li et al., 2026) | Newer, larger unified benchmark, 400K+ instances, 21 datasets, 33 models, per-prompt/per-model transparency | N/A (benchmark) | — | Own release | Multi-metric | <cite index="13-1">Provides a further related-work comparison noting RouterArena treats routing systems as black boxes and uses different model pools across routers, undermining cross-method comparability, and does not provide per-prompt, per-model data</cite> — this benchmark fixes that | Very recent, ecosystem/tooling still maturing | Good candidate as your **evaluation backbone** since it standardizes the model pool |
| **RouterEval** (Huang et al., 2025) | Massive-scale benchmark, >200M performance records across 8,500+ LLMs, 12 benchmarks | N/A (benchmark) | — | Own release | m-way classification framing (m=3..1000) | Shows **capable routers can beat the single best model** through complementarity — a strong motivating result | No cost metadata at all (accuracy-only) | Good for accuracy-only ablations, not for cost-aware work |
| **GraphRouter** (ICLR 2025) | Graph-based representation of query–model compatibility | GNN over a query-model bipartite graph | Graph embeddings | Multiple | Accuracy/cost | Captures relational structure between queries and models | Heavier ML machinery, less interpretable | Not evaluated under distribution shift / new models arriving over time |
| **RouterDC** (Chen et al., NeurIPS 2024) | Dual contrastive learning to build a query-based router | Contrastive embedding router | Query embeddings | Multiple | Accuracy/cost | Strong empirical results on RouterArena spider-plot ("excels in Cost-ratio Score") | No online learning | — |
| **CP-Router** (Su et al., 2025) | Uncertainty-aware routing between an LLM and a large reasoning model (LRM) using conformal prediction | <cite index="7-1">Applies conformal prediction to provide statistically rigorous uncertainty estimates for routing decisions</cite> | Conformal nonconformity scores | Reasoning benchmarks | Coverage-calibrated accuracy | First to bring **statistically valid** uncertainty guarantees into routing, rather than heuristic confidence | <cite index="7-1">CP has been applied broadly to NLU tasks but its application specifically to routing remains largely unexplored</cite> beyond this one paper | Conformal guarantees under **non-stationary** model pools (models get updated/deprecated) is untouched |
| **UniRoute** (Jitkrittum et al., 2026) | <cite index="43-1">Addresses dynamic routing where previously unseen LLMs become available at test time, representing each model as a feature vector derived from its predictions on representative prompts</cite> | Feature-vector model representation, cost-curve sweep | Model "fingerprint" vectors | Multiple | Cost-accuracy curve | Directly tackles the **new-model-arrival** problem | Sweeps a monetary-cost/λ curve rather than satisfying a hard SLO | Doesn't jointly handle latency SLOs — addressed partially by "Cluster, Route, Escalate" below |
| **Cluster, Route, Escalate** (2606.27457, 2026) | Two-stage: pre-route hard queries directly to strong model under a TPOT (time-per-output-token) budget, then a lightweight QE cascade for accuracy recovery | Clustering + explicit latency-budget routing + escalate-on-low-confidence | TPOT budget, correctness classifier | Custom | Cost @ SLO, accuracy | Explicit **hard latency SLO** satisfaction — most routing papers ignore tail latency | Complexity of two-stage pipeline; limited task diversity tested | Combining **carbon budget + latency SLO simultaneously** is still absent |
| **GAR — Green-Aware Routing** (2605.11603, 2026) | <cite index="24-1">Constrained multi-objective optimization that minimizes per-request CO2 emissions subject to explicit accuracy floors and p95-latency SLOs</cite> | Primal-dual online routing (GAR-PD) + heuristic variants | Correctness/latency/carbon estimators | Standard NLP benchmarks, heterogeneous 7B–70B pool | CO2 saved, accuracy floor violation rate, p95 latency | First to treat **carbon** as a first-class routing objective with formal constraints | <cite index="24-1">Current routing methods rarely consider sustainable energy use and CO2 emissions as optimization objectives, despite grid carbon intensity varying by time/region</cite> — this paper is one of the very few addressing it, so the space is still thin | Almost no work combines carbon-awareness with **personalization** or **multi-agent** settings — wide open |
| **OrcaRouter** (2605.30736, 2026) | Production router: LinUCB contextual bandit + offline ridge-regression warm start, online updates only on the *selected* arm | Hybrid offline (full-info) + online (bandit, partial-info) learning | Lexical + sentence-embedding features | RouterArena + RouterBench (5,000 filtered prompts) | Arena score, accuracy@cost | Realistic **partial-information online learning** (you only observe reward for the model you picked) — this is the correct real-world setting, most academic routers assume full-information | Only updates the chosen arm, so under-explored arms decay slowly; fixed 10-model pool | **Cold-start** and **arm-exploration efficiency** under partial feedback is a rich, tractable gap for a student project |
| **Watt Counts** (2604.09048, 2026) | Large energy-consumption dataset: 50 models × 10 GPUs, batch + server scenarios | N/A (dataset/benchmark) | — | Own release | Joules/token, throughput | Enables **hardware-aware routing research without owning a GPU cluster** — you can route against *their* measured energy table | Snapshot in time, hardware set is fixed | Perfect as a *plug-in energy cost table* for a student router (huge feasibility win) |

*(This is a representative slice, not the full field — dozens more exist, e.g. Zooter, AutoMix, MixLLM, TensorOpera Router, IRT-Router, HAPS, Meta-Router, Causal LLM Routing, ICL-Router, Firewall Routing, Self-REF, "Route to Reason." All were touched during search and are consistent with the gap analysis below.)*

### B. GitHub landscape

| Repo | Stars (approx.) | Last active | Problem solved | Missing / gap | Research-extendable? |
|---|---|---|---|---|---|
| **lm-sys/RouteLLM** | <cite index="33-1">~3.9k, Apache-2.0, from UC Berkeley/Anyscale/Canva, ICLR 2025</cite> | Actively maintained | Drop-in OpenAI-client-compatible binary router, ships pretrained routers | <cite index="36-1">Binary (strong/weak) only</cite>; no cost-SLO or energy objective; no online adaptation | **Yes** — best base to fork for a 3+-model, multi-objective extension |
| **vllm-project/semantic-router** | Growing fast, backed by vLLM project, active 2026 releases | Very active (v0.3 released mid-2026) | <cite index="37-1">Programmable Mixture-of-Models router that evaluates request signals, preferences and policies to select or compose a model path, adding safety/jailbreak/PII detection to routing</cite> | Focused on serving infra correctness/safety, not on cost-optimal or carbon-aware objectives; heavier infra to stand up | Partially — good as the **serving layer** under a novel routing policy |
| **ulab-uiuc/LLMRouter** | Crossed 1k★ in Jan 2026 | Active | <cite index="38-1">Unified open-source LLM routing library with 16+ routing strategies, CLI, Gradio UI, 11 datasets</cite> | <cite index="38-1">Their own roadmap explicitly lists: improve personalized routers (user profiling, cold-start), multimodal routing, and continual/online learning to adapt to domain drift as open items</cite> | **Excellent** — this is a maintained library whose *own maintainers* list your feasible research gaps as open TODOs |
| **LiteLLM** | Large, mainstream | Very active | OpenAI-compatible proxy/gateway over 100+ models, load balancing, spend tracking | <cite index="35-1">Primarily a proxy, not an intelligent router — task classification and routing logic must be built on top</cite> | Use as **infrastructure**, not as the research contribution itself |
| **RouterArena repo** | New (2026) | Active, live leaderboard | Standardized router evaluation harness | Own robustness issues (see judge-artifact findings above) | **Yes** — building a robustness/leaderboard-auditing tool is itself a viable paper |

**Takeaway for you:** `lm-sys/RouteLLM` and `ulab-uiuc/LLMRouter` are the two best forkable codebases — the latter's maintainers have literally published the research gaps (personalization, multimodal routing, online/continual adaptation) that align with "genuinely novel but feasible."

---

## Step 3: Research Landscape — Dimension Coverage Matrix

| Dimension | RouteLLM | FrugalGPT | Hybrid LLM | RouterDC/GraphRouter | CP-Router | GAR (carbon) | OrcaRouter (bandit) | UniRoute |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Cost optimization | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ✅ | ✅ |
| Latency optimization | ◐ | ◐ | ✗ | ✗ | ✗ | ✅(p95 SLO) | ◐ | ✗ |
| Accuracy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅(floor) | ✅ | ✅ |
| Energy efficiency | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ | ✗ | ✗ |
| Carbon awareness | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ | ✗ | ✗ |
| Uncertainty/confidence estimation | ✗ | ◐(scorer) | ✗ | ✗ | ✅(conformal) | ◐ | ✗ | ✗ |
| Online/continual learning | ✗ | ✗ | ✗ | ✗ | ✗ | ✅(primal-dual) | ✅(bandit) | ✗ |
| Personalization | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Retrieval-awareness (RAG-aware routing) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Context-length awareness | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Hardware-awareness | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | ✗ | ✗ |
| Multi-agent compatibility | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Dynamic model-pool adaptation (new/retired models) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | ✅ |
| Evaluation-artifact robustness | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

**Empty/near-empty columns = your opportunity space.** Personalization, retrieval-awareness, context-length-awareness, and multi-agent compatibility are almost completely unaddressed as *combined* dimensions (a few papers touch one in isolation, but no paper combines even two of {carbon, personalization, retrieval-awareness, evaluation-robustness}).

---

## Step 4: Genuine Research Gaps (not "add another feature")

1. **No router jointly optimizes carbon + latency-SLO + accuracy under partial (bandit) feedback.** GAR does carbon+accuracy+latency but assumes full-information offline evaluation; OrcaRouter does partial-feedback bandit routing but ignores carbon entirely. Combining them is a genuine, unexplored intersection, not just "add a feature" — it changes the optimization structure (constrained bandit vs. unconstrained bandit).
2. **Evaluation-artifact robustness is itself under-researched.** RouterArena's own follow-up work found up to 24-point judge-scoring bias and 65% response truncation distorting reported "routing headroom." A router benchmark that is **robust to grading-method choice** (dual-judge + exact-match reconciliation, truncation-safe evaluation) is a methodological gap with clear IEEE/workshop publication value and needs zero GPU training.
3. **Cold-start / new-model-arrival under partial feedback.** UniRoute solves cold-start with full-information probing; OrcaRouter solves partial-feedback but with a fixed model pool. No paper solves "a new model appears, and you may only observe reward on models you actually route to" — a classic **contextual bandit with a growing action space**, tractable with vLLM/OpenRouter APIs on a student budget.
4. **Personalization is essentially absent** despite `ulab-uiuc/LLMRouter`'s own maintainers flagging "personalized routers: stronger user profiling, cold-start strategies" as an open TODO. A router that adapts its cost/quality tradeoff to a *specific user's* revealed preferences (not just a global average) over a session/history is unexplored and matches a bandit-with-user-context formulation.
5. **Retrieval-awareness / RAG-aware routing** — no paper studies routing at the granularity of "does this query need retrieval, and if so, does the *retrieval quality* change which model should answer." This directly matters for the huge fraction of real deployments that are RAG pipelines, and lets you reuse cheap, small embedding models rather than large ones.
6. **Weak assumption:** almost all routers assume a **static, fixed model pool during evaluation** — real deployments see models deprecated/repriced weekly (as literally shown by OrcaRouter's own "pricing snapshot used at submission time" caveat). A router evaluated under **simulated pool churn** (models appear/disappear/reprice over an evaluation window) is a realistic, easy-to-simulate, unaddressed setting.
7. **Missing evaluation metric:** nobody reports **routing regret decomposition** — i.e., how much of the accuracy/cost gap to an oracle router comes from (a) bad model-pool coverage, (b) router mis-calibration, vs. (c) irreducible task ambiguity. RouterArena's "unsolvability ceiling" work is the closest thing, but doesn't decompose regret by *routing-system* fault vs. *benchmark* fault.
8. **Missing deployment setting:** **on-device / edge routing where the router itself must run on constrained hardware** (the "Sustainability-Aware LLM Inference on Edge Clusters" line of work) touches this but doesn't combine it with learned, adaptive (not static greedy) routing policies.

### Gap ranking (Novelty / Practicality / Engineering complexity / Publication potential, all /10)

| Gap | Novelty | Practicality (for 6mo undergrad) | Eng. complexity | Publication potential |
|---|---|---|---|---|
| 1. Carbon + latency-SLO + bandit routing | 8 | 7 | 6 | 8 |
| 2. Evaluation-artifact robustness / benchmark auditing | 7 | 9 | 3 | 7 |
| 3. Cold-start under partial feedback, growing action space | 7 | 8 | 5 | 7 |
| 4. Personalized routing | 6 | 8 | 4 | 6 |
| 5. Retrieval-aware routing | 7 | 8 | 5 | 7 |
| 6. Pool-churn robustness simulation | 8 | 8 | 4 | 7 |
| 7. Regret decomposition metric | 7 | 9 | 3 | 6 |
| 8. Edge-constrained adaptive router | 6 | 6 | 7 | 6 |

---

## Step 5: 15+ Novel Project Ideas

Each entry is deliberately scoped to Python + HF + vLLM/LiteLLM/OpenRouter + Colab/consumer GPU, 4–6 months.

**1. CarbonBandit — Carbon-and-Latency-Constrained Contextual Bandit Router**
- *Motivation:* No router combines carbon-awareness (GAR) with realistic partial-feedback bandit learning (OrcaRouter).
- *Why current work misses it:* GAR assumes offline full-information reward matrices; OrcaRouter ignores carbon.
- *Hypothesis:* A constrained LinUCB/Thompson-sampling bandit with a carbon-budget constraint (using Watt Counts' measured energy table + a public grid-carbon-intensity API) can match GAR's carbon reduction while learning online from partial feedback.
- *Novel contribution:* First carbon-constrained *partial-feedback* router; reusable "energy cost table" plug-in from Watt Counts removes the need to measure GPU power yourself.
- *Architecture:* Feature extractor (sentence embeddings) → constrained contextual bandit (primal-dual LinUCB) → model pool via LiteLLM/OpenRouter.
- *Algorithms:* Constrained contextual bandits (primal-dual UCB), ridge regression reward model.
- *Datasets:* RouterBench (precomputed, cheap), Watt Counts (energy), a public grid carbon-intensity API (e.g., Electricity Maps free tier) for the carbon signal.
- *Baselines:* RouteLLM, GAR-Fixed, plain LinUCB (no carbon constraint).
- *Metrics:* Accuracy, $/1k queries, CO2/1k queries, SLO-violation rate, cumulative regret.
- *Challenges:* Getting a real-time carbon-intensity signal without a paid API (use daily-average public data as a proxy — acceptable for a student project).
- *Publication potential:* High — clean gap, workshop/IEEE-conference scoped.

**2. RouterArena-Robust — A Grading-Robust Router Evaluation Toolkit**
- *Motivation:* RouterArena's own follow-ups show judge-scoring bias up to 24 points and truncation affecting 65% of responses.
- *Why missed:* Existing benchmarks report a single LLM-judge score; nobody audits *how much of the routing "win" is scoring artifact*.
- *Hypothesis:* Reconciling LLM-judge scores with exact-match/rubric scores where possible will shrink the apparent routing headroom substantially, and this shrinkage differs systematically by task domain.
- *Novel contribution:* A reusable "dual-judge + truncation-aware" evaluation wrapper other students/researchers can drop onto any router.
- *Architecture:* Evaluation harness sitting on top of RouterBench/RouterArena-style data; dual scorer (exact-match where gold labels exist + LLM-judge elsewhere) + truncation-rate reporting.
- *Algorithms:* None novel required — this is a rigorous empirical-methodology contribution (very safe for undergrads: **zero training needed**).
- *Datasets:* RouterBench, MMLU/MedQA subsets with gold answers.
- *Baselines:* Reported numbers from RouteLLM, GraphRouter, RouterDC on RouterArena's leaderboard.
- *Metrics:* Score deviation (judge vs. exact-match), truncation rate, corrected Arena-score deltas.
- *Challenges:* Needs careful experimental design, not engineering; risk of being seen as "incremental" unless framed sharply — mitigate by producing a reusable open-source tool + concrete corrected leaderboard.
- *Publication potential:* High for a systems/empirical-methods workshop (very undergrad-friendly: cheap, fast, rigorous).

**3. ColdRoute — Cold-Start Routing for a Growing Model Pool**
- *Motivation:* UniRoute solves cold-start with full-information; real deployments only observe reward on the arm chosen.
- *Hypothesis:* A meta-learned "model fingerprint" (à la UniRoute) can warm-start a partial-feedback bandit (à la OrcaRouter) so that a newly added model reaches near-optimal routing share within N queries instead of requiring full probing.
- *Novel contribution:* Combines UniRoute's model-fingerprint idea with OrcaRouter's bandit — a genuinely new hybrid, not a re-implementation of either.
- *Architecture:* Offline fingerprint extractor (predictions on a fixed probe set) → bandit initialization → online LinUCB updates.
- *Algorithms:* Feature-based bandit initialization, LinUCB/Thompson sampling.
- *Datasets:* RouterBench + LLMRouterBench (33 models — good testbed for "new model arrives" simulation).
- *Baselines:* Cold LinUCB (no warm start), UniRoute (full-info only), OrcaRouter.
- *Metrics:* Regret vs. number of queries since model added, time-to-90%-optimal routing share.
- *Challenges:* Simulating "arrival" realistically; keep pool size small (5–8 models) to stay in budget.
- *Publication potential:* High — directly extends two 2026 papers with a clear, testable hypothesis.

**4. PersonaRoute — User-Personalized Cost/Quality Router**
- *Motivation:* `ulab-uiuc/LLMRouter`'s own maintainers list personalization as an open item.
- *Hypothesis:* Different users have different revealed cost/quality tradeoffs (a student vs. a paying professional); a per-user contextual bandit that adapts the quality threshold from implicit feedback (accept/regenerate/edit) will outperform a single global router.
- *Novel contribution:* First router with an explicit **user-context arm** rather than only query-context.
- *Architecture:* Query embedding ⊕ user-history embedding → contextual bandit.
- *Algorithms:* Contextual bandit with user features, simple collaborative-filtering-style prior.
- *Datasets:* Simulate users by sampling different implicit "quality thresholds" over RouterBench queries (a well-accepted simulation methodology in bandit papers).
- *Baselines:* RouteLLM (single global router), per-user independent bandits (no sharing).
- *Metrics:* Cost saved per user at matched user-satisfaction proxy, cold-start speed per new user.
- *Challenges:* No real user-preference dataset exists — must simulate, and should be transparent about that limitation in the writeup.
- *Publication potential:* Medium-high; simulation-based personalization is a known, accepted methodology at workshops.

**5. RAG-Router — Retrieval-Aware Model Selection**
- *Motivation:* No existing router conditions on retrieval quality/necessity.
- *Hypothesis:* Query complexity *after* retrieval (i.e., how much the retrieved context actually helps) is a better routing feature than query complexity alone; many "hard-looking" queries become easy once good context is retrieved.
- *Novel contribution:* Two-stage router: (a) decide if retrieval helps, (b) route to model size conditioned on retrieval outcome.
- *Architecture:* Retriever (BM25/dense) → context-quality scorer → router.
- *Algorithms:* Lightweight classifier on (query, retrieved-context similarity, retrieval confidence).
- *Datasets:* RAG benchmarks (e.g., HotpotQA, Natural Questions subsets) + your own retrieval corpus.
- *Baselines:* RouteLLM applied naively on top of RAG, "always retrieve + large model."
- *Metrics:* Accuracy, cost, retrieval-call count.
- *Challenges:* Building a decent retrieval pipeline adds scope — keep the corpus small (Wikipedia subset).
- *Publication potential:* Medium-high, very relevant to industry (most production LLM apps are RAG).

**6. ChurnRobust — Routing Under Simulated Model-Pool Churn**
- *Motivation:* All current benchmarks assume a static pool; real APIs reprice/deprecate weekly.
- *Hypothesis:* Routers trained/evaluated only on a static pool overfit to specific models' price/quality points and degrade sharply when prices shift or a model is silently swapped for a new version.
- *Novel contribution:* First systematic "pool-churn robustness" benchmark + a router regularization technique that stays robust to reasonable price/quality perturbations.
- *Architecture:* Wrap RouterBench with a churn-simulator (randomly perturb price/accuracy by realistic amounts drawn from historical OpenRouter pricing changes) → evaluate existing routers' robustness → propose a robustness-regularized router.
- *Algorithms:* Distributionally-robust optimization (lightweight version, e.g. min-max over perturbation set) applied to a simple router.
- *Datasets:* RouterBench + historical OpenRouter pricing snapshots (public).
- *Baselines:* RouteLLM, GAR, un-regularized bandit.
- *Metrics:* Accuracy/cost degradation under churn, robustness margin.
- *Challenges:* Defining a realistic churn distribution — justify with real historical pricing data.
- *Publication potential:* High novelty, practical relevance, moderate engineering.

**7. RegretLens — Routing Regret Decomposition Framework**
- *Motivation:* No metric separates "router is bad" from "benchmark/model-pool makes the task unsolvable."
- *Hypothesis:* Decomposing the gap-to-oracle into (pool-coverage regret, calibration regret, irreducible-ambiguity regret) will show that published routers differ mainly on calibration regret, not pool coverage — actionable for future router design.
- *Novel contribution:* A reusable regret-decomposition metric/toolkit (methodological, not a new router).
- *Architecture:* Statistical analysis pipeline over existing precomputed benchmark outputs (RouterBench/LLMRouterBench) — no training needed.
- *Algorithms:* Oracle computation, regret decomposition via counterfactual analysis.
- *Datasets:* RouterBench, LLMRouterBench (has per-prompt per-model data — required for this).
- *Baselines:* Reported aggregate accuracy/cost numbers from existing papers.
- *Metrics:* The decomposition itself is the output.
- *Challenges:* Needs careful, defensible statistical methodology.
- *Publication potential:* Medium-high, very low compute cost, strong "critical analysis" angle reviewers like.

**8. EdgeRoute-Lite — Adaptive Router for Resource-Constrained Edge Devices**
- *Motivation:* Existing edge/carbon routing work uses static greedy rules, not learned adaptive policies.
- *Hypothesis:* A tiny (<10M param) learned router can outperform greedy carbon/latency heuristics on Jetson-class hardware while adding negligible overhead.
- *Novel contribution:* Learned-vs-greedy comparison specifically on edge hardware constraints.
- *Architecture:* Distilled tiny classifier router + local small model + cloud large model fallback.
- *Algorithms:* Knowledge distillation of a larger router into a tiny classifier.
- *Datasets:* RouterBench (for router training), your own timing/power measurements on Colab/Jetson-equivalent if available, else use Watt Counts numbers as proxy.
- *Baselines:* Greedy latency/carbon routing (from the "Sustainability-Aware Edge" line of work).
- *Metrics:* Accuracy, latency, energy, router-overhead.
- *Challenges:* No physical Jetson hardware → must proxy with published measurements (state clearly as a limitation) or use a laptop-CPU vs. cloud-GPU split as an accessible substitute.
- *Publication potential:* Medium (compute-access risk is the main constraint).

**9. MultiObjectiveRoute — Pareto-Front Router with User-Tunable Preferences**
- *Motivation:* Most routers optimize a single scalarized objective; users actually want to *move along* a cost/latency/quality Pareto front.
- *Hypothesis:* Training a router to predict the full Pareto front (not a single point) lets a user pick their tradeoff at inference time without retraining.
- *Novel contribution:* Pareto-front prediction instead of scalarized routing decision.
- *Architecture:* Multi-output regressor predicting (accuracy, cost, latency) per candidate model; a simple selection rule picks the Pareto-optimal model given a user weight vector.
- *Algorithms:* Multi-task regression, Pareto-front computation.
- *Datasets:* RouterBench, LLMRouterBench.
- *Baselines:* RouteLLM (single scalarized threshold).
- *Metrics:* Hypervolume of predicted vs. true Pareto front, user-satisfaction proxy across sampled preference weights.
- *Challenges:* Multi-output regression calibration.
- *Publication potential:* Medium-high, clean ML contribution.

**10. SafeRoute — Safety/Jailbreak-Aware Cost Routing**
- *Motivation:* vLLM Semantic Router adds safety filtering but as a binary gate, not integrated into the cost-optimization objective.
- *Hypothesis:* Treating "risk of harmful/jailbreak content" as a routing feature (route risky-looking queries to models with stronger safety tuning, not just the cheapest capable model) improves the safety/cost Pareto front vs. a separate safety gate.
- *Novel contribution:* Joint cost-safety objective rather than sequential "filter then route."
- *Architecture:* Lightweight jailbreak/toxicity classifier feeding into the router's feature vector.
- *Algorithms:* Standard classifier + routing policy combination.
- *Datasets:* Public jailbreak/safety benchmarks (e.g., a subset of open safety-eval sets) + RouterBench for quality/cost.
- *Baselines:* vLLM Semantic Router's "filter-then-route" design.
- *Metrics:* Safety-violation rate, cost, accuracy.
- *Challenges:* Handling safety data responsibly and narrowly (use only public, well-established eval sets — do not generate new harmful content).
- *Publication potential:* Medium; safety-adjacent framing needs care but is a real, underexplored axis.

**11. ContextRoute — Context-Length-Aware Routing**
- *Motivation:* No paper routes based on *context-window cost scaling* (long-context calls are disproportionately expensive on some models but not others).
- *Hypothesis:* For long-context queries, model choice should weight $/token-at-this-length rather than a flat per-query cost estimate, changing routing decisions substantially for RAG/long-document tasks.
- *Novel contribution:* Length-conditioned cost model plugged into an existing router (e.g., RouteLLM) as an ablation-friendly add-on.
- *Architecture:* Cost estimator conditioned on (query length, expected output length) → router.
- *Algorithms:* Simple parametric cost curves fit from public pricing data.
- *Datasets:* Long-document QA sets (e.g., a subset of NarrativeQA or GovReport) + RouterBench for short-query baseline comparison.
- *Baselines:* Flat-cost RouteLLM.
- *Metrics:* Cost savings specifically on long-context subset.
- *Challenges:* Needs realistic long-context test data; keep dataset small.
- *Publication potential:* Medium, very practical/industry-relevant.

**12. FeedbackRoute — Implicit-Feedback Online Router (No Labels Needed)**
- *Motivation:* Most routers assume labeled correctness; real deployments only have implicit signals (user re-asks, edits, thumbs down).
- *Hypothesis:* A router trained purely on implicit signals (regeneration rate, edit distance between turns) can approximate a fully-supervised router's cost savings within a small gap.
- *Novel contribution:* Weak-supervision routing from implicit signals only.
- *Architecture:* Simulate implicit feedback from existing labeled benchmarks (treat "wrong answer" as triggering a simulated "user re-asks"), train bandit on this weak signal.
- *Algorithms:* Weakly-supervised contextual bandit.
- *Datasets:* RouterBench (simulate implicit signals from known-correct labels).
- *Baselines:* Fully-supervised RouteLLM, random routing.
- *Metrics:* Gap-to-fully-supervised cost/accuracy.
- *Challenges:* Simulation fidelity — be explicit this is a simulation study.
- *Publication potential:* Medium-high, very relevant to real deployment constraints.

**13. AgentRoute — Per-Step Routing Inside Multi-Agent/Tool-Use Pipelines**
- *Motivation:* "Multi-agent compatibility" is an almost entirely empty column in the landscape matrix.
- *Hypothesis:* Different steps of an agentic pipeline (planning vs. tool-call formatting vs. final synthesis) have very different difficulty/cost profiles; a step-level router beats using one model for the whole agent loop.
- *Novel contribution:* Extends routing from single-turn queries to **per-step routing within an agent trajectory**, a genuinely different problem (steps are correlated, not i.i.d.).
- *Architecture:* Agent framework (e.g., a simple ReAct loop built with LangChain/LiteLLM) with a router at each step type.
- *Algorithms:* Step-type classifier + per-type router; simple credit-assignment heuristic for trajectory-level reward.
- *Datasets:* A small agentic benchmark (e.g., a subset of tool-use tasks like ToolBench or a self-built small task set).
- *Baselines:* Single-model agent, query-level (not step-level) router applied naively.
- *Metrics:* Task success rate, total cost per successful trajectory.
- *Challenges:* Building even a small agent harness adds real engineering scope — keep tool-set small (3–4 tools).
- *Publication potential:* High novelty (genuinely empty gap), moderate-high engineering complexity — good "reach" option if the team is strong.

**14. CascadeOrder — Learned (Not Fixed) Cascade Ordering**
- *Motivation:* FrugalGPT-style cascades probe models in a **fixed** cheap-to-expensive order; "Brick" and "Cluster-Route-Escalate" show ordering matters a lot for cost overhead from rejected calls.
- *Hypothesis:* Learning a query-dependent probe order (not always cheapest-first) reduces wasted "rejected-call" cost versus a fixed-order cascade, especially when the cheapest model is *not* actually most likely to succeed on a given query type.
- *Novel contribution:* Query-conditional cascade ordering, not just a query-conditional accept/reject threshold.
- *Architecture:* A ranking model predicts the expected-utility probe order per query, then cascades along that order.
- *Algorithms:* Learning-to-rank applied to cascade ordering.
- *Datasets:* RouterBench, LLMRouterBench.
- *Baselines:* FrugalGPT (fixed cheap-first order), Cascade Routing.
- *Metrics:* Amortized cost including rejected-call overhead, accuracy.
- *Challenges:* Correctly accounting for "wasted" cost of rejected probes (the Brick paper shows this matters a lot).
- *Publication potential:* Medium-high, clean and well-scoped extension of a well-known baseline.

**15. RouterBench-Small — Data-Efficient Router Training (Few-Shot Routing)**
- *Motivation:* Nearly all routers assume thousands of labeled preference/correctness examples for training; real teams deploying routers for a niche domain (e.g., legal, medical) won't have that.
- *Hypothesis:* A router meta-trained across many domains in RouterBench/LLMRouterBench can adapt to a *new* narrow domain with only tens-to-hundreds of examples (few-shot), matching a fully-trained-per-domain router.
- *Novel contribution:* Meta-learning / transfer-learning framing for router training data-efficiency — directly matches the undergrad constraint of limited API budget.
- *Architecture:* Meta-train router across domains in LLMRouterBench (21 datasets) → fine-tune/adapt on a held-out domain with small samples.
- *Algorithms:* Few-shot fine-tuning or in-context routing (prompt-based router, no training needed for the ICL variant).
- *Datasets:* LLMRouterBench (21 datasets → natural meta-learning splits).
- *Baselines:* Domain-specific router trained from scratch with same small sample budget.
- *Metrics:* Accuracy/cost vs. number of adaptation examples.
- *Challenges:* Needs careful domain-holdout design to avoid leakage.
- *Publication potential:* High relevance (directly useful to industry), clean experimental design, and the *lowest API-cost* idea on this list — very good fit for a constrained student budget.

**16. (Bonus) LatencyHonest — Wall-Clock-Faithful Router Evaluation**
- *Motivation:* Almost all routing papers report *predicted* cost/latency from pricing tables, not *measured* end-to-end latency including provider queueing/variance.
- *Hypothesis:* Real measured latency variance (especially p95/p99 tail) changes which router "wins" compared to using only advertised per-token latency.
- *Novel contribution:* A measurement-based (not table-based) latency evaluation layer.
- *Architecture:* Wrap OpenRouter/LiteLLM calls with real timing instrumentation across time-of-day, replay against existing router policies.
- *Algorithms:* None novel — empirical measurement + statistical analysis.
- *Datasets:* Live API calls on a small RouterBench subset (keep call count modest — a few hundred, well within student API budgets).
- *Baselines:* Any router's own "expected latency" claims.
- *Metrics:* Measured vs. claimed latency gap, tail-latency-adjusted ranking changes.
- *Challenges:* Requires live API access (budget a small amount for real calls — a few hundred, not thousands).
- *Publication potential:* Medium, but a very fast, cheap, honest empirical contribution.

---

## Step 6: Feasibility Filter (applied)

All 16 ideas above already satisfy the constraint (Python/HF/vLLM/LiteLLM/OpenRouter/Colab, no proprietary data, no massive training). Ideas **#2, #7, #15, #16** are the *cheapest* (near-zero or minimal new API calls, reuse precomputed benchmarks). Ideas **#8, #13** carry the highest engineering/infrastructure risk (hardware access, agent-harness building) and should only be picked by a stronger team with slack time.

## Step 7: Extension Opportunities (for top papers)

- **RouteLLM → extend by:** adding a 3rd+ model tier and a carbon/latency objective (feeds ideas #1, #6, #9). Existing GitHub repo (`lm-sys/RouteLLM`) removes most infra work — expected improvement: multi-objective Pareto coverage RouteLLM structurally cannot express.
- **FrugalGPT + Cascade Routing → combine into:** learned cascade *ordering* (idea #14) instead of fixed cheap-first order; expected improvement is reduced amortized cost from rejected-call overhead (the "Brick" paper already quantifies this overhead — reuse their accounting method).
- **UniRoute + OrcaRouter → combine into:** cold-start-aware partial-feedback bandit (idea #3); new objective = minimize regret specifically during the first N queries after a new model is added — a metric neither paper reports today.
- **RouterArena + LLMRouterBench → combine into:** the robust, per-prompt-transparent evaluation toolkit (idea #2/#7) — LLMRouterBench's per-prompt data plus RouterArena's known judge-artifact findings give you both the tool and the motivating evidence in one place.
- **GAR + OrcaRouter → combine into:** idea #1 (carbon-constrained bandit) — expected improvement: matching GAR's carbon reduction while relaxing its full-information assumption, which is the realistic deployment constraint OrcaRouter already validated as necessary.

## Step 8: Final Recommendation

### Top 5, scored

| # | Idea | Novelty | Feasibility | Publication Potential | Implementation Difficulty (lower=easier, shown as /10 difficulty) | Expected Impact |
|---|---|---|---|---|---|---|
| 1 | CarbonBandit | 8 | 7 | 8 | 6 | 8 |
| 3 | ColdRoute | 7 | 8 | 7 | 5 | 7 |
| 2 | RouterArena-Robust | 7 | 9 | 7 | 3 | 7 |
| 15 | RouterBench-Small (few-shot routing) | 7 | 9 | 8 | 4 | 8 |
| 14 | CascadeOrder | 6 | 8 | 6 | 4 | 6 |

### Final single recommendation: **Idea #15 — RouterBench-Small (Data-Efficient / Few-Shot Router Adaptation)**, optionally paired with elements of Idea #2 (robust evaluation) as a secondary contribution.

**Why this is the best balance for an undergraduate team:**
- **Feasibility:** Uses only precomputed benchmark data (LLMRouterBench, RouterBench) — near-zero live API spend, no GPU training beyond a small classifier/adapter, fits comfortably in Colab.
- **Novelty:** Directly targets a documented gap (`ulab-uiuc/LLMRouter`'s maintainers flag "cold-start strategies" as unsolved; no paper studies few-shot/meta-learned router adaptation to a new narrow domain).
- **Publication potential:** The result is immediately useful to practitioners (most companies deploying routers have a niche domain with little labeled data) — a strong "practical relevance" hook for an IEEE-track paper.
- **Why it's still novel despite existing literature:** RouteLLM, Hybrid LLM, GraphRouter, RouterDC all assume a large, in-domain training set is available; UniRoute solves cold-start for *new models*, not new *domains*; ulab-uiuc/LLMRouter names domain adaptation as future work rather than solving it. No paper found in this search combines meta-learning-style transfer with the specific router-training problem.
- **6-month timeline:** Months 1–2 literature/dataset setup + reproducing a baseline router; Months 2–4 meta-learning/few-shot adaptation implementation and ablations; Months 4–5 writing + robustness checks (optionally borrowing idea #2's dual-scoring methodology to strengthen result credibility); Month 6 paper writing/submission prep.

---

### Suggested next step

The above is grounded in real, current sources, but Steps 1–2 as you specified them (exhaustive NeurIPS/ICML/ICLR/ACL/EMNLP/NAACL/AAAI/IEEE/ACM sweep + full GitHub audit with stars/last-update for every repo) genuinely benefits from a longer, multi-source background pass rather than a single chat turn — that would let me pull 20–40 more papers/repos, verify star counts and last-commit dates precisely, and build the complete comparison table with citations for every cell.
