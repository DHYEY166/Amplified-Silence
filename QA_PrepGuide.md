# Q&A Preparation Guide — Amplified Silence (DSCI 531)

---

## Part 1 — Glossary of Key Terms

### ALS (Alternating Least Squares)
A collaborative filtering algorithm for implicit-feedback recommendation systems. It factorizes the user–item interaction matrix into two low-dimensional matrices (user factors and item factors) by alternately fixing one and solving for the other in a least-squares sense.
- **Why we used it:** ALS is the standard baseline for implicit-feedback datasets (play counts, streams) where you don't have explicit ratings. It is fast, well-studied, and the `implicit` library provides a GPU-ready implementation.
- **Our settings:** 64 latent factors, 20 iterations, regularization λ = 0.08.

### Implicit Feedback
User behaviour signals that are not explicit ratings (e.g., play counts, clicks, watch time). A user playing a song 50 times signals preference more strongly than playing it once, but never explicitly said "I like this." We log-scale the play counts and use them as *confidence weights* in ALS, not as ratings.

### Collaborative Filtering
Recommends items based on patterns of similar users — "users who listened to A also listened to B." No content features (genre, lyrics) are used; only the user–item interaction matrix.

### Candidate Pool (pool = 400)
ALS scores every artist for each user. We first retrieve the top-400 highest-scoring artists as a *candidate pool*, then apply fairness re-ranking within that pool to produce the final top-10. This avoids re-running the full model for every fairness scenario.

### Provider-Side Fairness
Fairness measured from the *artist's* perspective — do female artists get as much algorithmic visibility as male artists? Contrasts with *consumer-side fairness* (are recommendations equally good for all user groups).

### Exposure Male Share (position-weighted)
The fraction of total recommendation exposure that goes to male artists, weighted by rank position:
- **Weight at rank r** = 1 / log₂(r + 1) — items at rank 1 get the most weight.
- **Formula:** Σ(weight × male_item) / Σ(weight × labeled_item)
- **Baseline value:** 69.8% — meaning ~70% of all weighted exposure goes to male artists.
- **Parity = 50%.**

### Avg Rank Gap (M − F)
Mean rank of male artists in recommendations minus mean rank of female artists. A positive value means male artists appear higher (better ranked) on average.
- Baseline: +0.090 (male artists ranked slightly higher on average)
- Mit B: +0.451 (female artists boosted up, males pushed down slightly)

### Coverage Gap (M − F)
Number of *distinct* male artists recommended (across all users) minus distinct female artists recommended. Measures catalog diversity.
- Baseline: 185 more distinct male artists recommended than female.
- Mit C: only 21 — the most equitable.

### Precision@10
Of the 10 artists recommended, what fraction were in the user's actual test set (held-out plays)?
- **Formula:** (# relevant in top-10) / 10
- **Our baseline:** 0.132 → about 1.3 out of 10 recommendations are "correct."

### Recall@10
Of all artists the user actually interacted with in the test set, what fraction appear in the top-10 recommendations?
- **Formula:** (# relevant in top-10) / (total relevant for user)
- Measures how much of the user's true interest is captured.

### NDCG@10 (Normalized Discounted Cumulative Gain)
A ranked quality metric that rewards relevant items appearing *higher* in the list.
- Relevant items at rank 1 score more than relevant items at rank 8.
- Normalized to [0, 1] against the ideal ranking.
- **Our baseline:** 0.1745. Mit B drops it to 0.1717 — only a 1.6% relative decrease.

### Sensitivity Sweep
We varied the female score boost parameter `b` across {0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45} and measured how NDCG@10 and exposure male share change together. This shows the *rate of tradeoff* — how much utility you sacrifice per unit of fairness gained.

### Bias Amplification
The phenomenon where a recommendation algorithm takes an already-skewed training dataset and makes the skew *worse* in its outputs.
- Training data: 72.8% male exposure.
- Baseline recommendations: 69.8% male exposure.
- Here, ALS slightly *reduces* the skew from training data, but the absolute gap (≈70%) is still large and the system perpetuates systemic underrepresentation of female artists.

### Mitigation A — Score Boost +12%
Post-processing step: multiply every female artist's ALS score by **1.12** before re-ranking the candidate pool into top-10. No model retraining. Result: male exposure share drops from 69.8% → 60.3% with negligible NDCG loss (−0.3%).

### Mitigation B — Score Boost +28%
Same idea but stronger multiplier: **1.28**. Result: male exposure drops to 49.2% (near parity at 50%) with only 1.7% NDCG loss. **Identified as the practical sweet spot.**

### Mitigation C — Constrained Re-ranking
Hard rule: force at least **4 female artists** into every top-10 list, selecting the highest-scoring female artists first, then filling remaining slots by score. Result: male exposure collapses to 14.6% (over-correction) and NDCG drops by 37% — too aggressive.

### MusicBrainz
An open music encyclopedia with structured metadata including artist type (person/group) and gender for person artists. We queried the MusicBrainz API at ~1 request/second (rate limited) to label each artist as male, female, or unknown. After 11,000 API calls: **3,867 artists labeled M/F** out of 10,362 total (37.3% coverage).

### Train/Test Split (per-user 80/20)
For each user, 80% of their interactions go into the training matrix (used to train ALS), and 20% are held out as the test set to evaluate recommendations. The split is done per-user to ensure every user has both training and evaluation data.

---

## Part 2 — Figure Explanations

### Figure 1: Provider-side Gender Exposure — Baseline vs. Mitigation Strategies
**What it shows:** A bar chart with one bar per model (Baseline, Mit A, Mit B, Mit C). Each bar's height is the **position-weighted male exposure share** — the fraction of total ranked exposure (top-10 lists, weighted by rank) that goes to male artists.

**Key reference lines:**
- **Dashed line at 72.8%** = male share in the raw training data (how skewed the input is).
- **Dotted line at 50%** = gender parity target.

**How to read it:** The baseline sits at 69.8% — close to the training skew, confirming ALS largely inherits it. Each mitigation moves the bar progressively toward parity. Mit B (49.2%) essentially reaches parity. Mit C (14.6%) overcorrects — male artists are now severely under-exposed.

**Key takeaway:** Score boosting smoothly dials down male overexposure; hard constraints overshoot parity.

---

### Figure 2: From Training Data to Recommendations — Gender Exposure at Each Stage
**What it shows:** A 4-bar pipeline chart tracing male exposure share across stages: Training Data → Baseline Recs → Mit A Recs → Mit B Recs.

**How to read it:** Each bar shows the male share at that point in the pipeline. The drop from bar to bar shows how much each post-processing step corrects the inherited skew.

| Stage | Male Share | Change |
|---|---|---|
| Training Data | 72.8% | — (input bias) |
| Baseline Recs | 69.8% | −3 pp (ALS slightly reduces but preserves skew) |
| Mit A (+12%) | 60.3% | −9.5 pp |
| Mit B (+28%) | 49.2% | −11.1 pp (near parity) |

**Key takeaway:** Bias is baked in at the data level and carried through to recommendations. Post-processing can progressively correct it without retraining.

---

### Figure 3: Fairness–Utility Trade-off as Female Boost Increases
**What it shows:** A scatter/line plot where each point represents one boost value `b` from the sensitivity sweep {0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45}.
- **X-axis:** Male exposure share (moving left = fairer, more female exposure).
- **Y-axis:** NDCG@10 (higher = better recommendation quality).
- Named points: Baseline (b=0), Mit A (b=0.12), Mit B (b=0.28) are highlighted.

**How to read it:** The curve starts at the top-right (Baseline — high NDCG, high male share) and moves left as the boost increases. Crucially, the curve is **nearly flat** (NDCG barely drops) until around b=0.28 (Mit B), then drops more steeply. This "elbow" in the curve marks the practical optimum.

**Key takeaway:** There is a "free zone" of fairness improvement — you can push male exposure from 70% down to ~50% while sacrificing less than 2% in NDCG. Beyond that, each additional fairness gain costs increasingly more utility.

---

### Figure 4 (in paper only): Sensitivity of NDCG vs. Boost
Same data as Figure 3 but plotted differently — boost value on X-axis vs. NDCG on Y-axis. Shows a clean monotonic decrease in NDCG as boost increases, confirming the tradeoff is smooth and predictable (no sudden cliff until b > 0.30).

---

## Part 3 — Expected Q&A

### Q1.  Why did you choose ALS over other recommendation algorithms like neural collaborative filtering or BPR?
**A:** ALS is the de facto standard for implicit-feedback collaborative filtering and serves as a strong, interpretable baseline. Our goal was to study fairness properties in a well-understood model before moving to black-box neural approaches. ALS also scales efficiently to our dataset size (24K users, 10K items, 1M interactions). Neural methods could be explored in future work.

---

### Q2. Your training data is 72.8% male but baseline recommendations are 69.8% male. Doesn't that mean ALS is actually reducing bias?
**A:** Slightly — but the absolute level of underrepresentation remains severe. Female artists still receive only ~30% of exposure in a world where they make up roughly half the artist population. The small reduction may also be a statistical artifact of how play counts are log-scaled. The key finding is the *persistence* and *structural embedding* of bias, not whether it worsens marginally.

---

### Q3. Why only post-processing mitigations? Why not retrain the model with fairness constraints?
**A:** Post-processing is practical — it requires no model retraining, can be applied to any existing recommender system, and is easy to tune at deployment time. In-processing constraints (modifying the ALS objective) are a natural next step but significantly more complex to implement and tune. We wanted to quantify what is achievable with the simplest possible intervention first.

---

### Q4. Why did Mitigation C collapse utility so much (−37% NDCG)?
**A:** Forcing ≥4 female artists into every top-10 list regardless of ALS scores means we sometimes surface artists the model rates very poorly for that user — they may not match the user's listening history at all. This hard constraint completely ignores personalization signals for those slots. Score boosting (Mit A/B) still respects the model's personalization while gently lifting female artists — hence much smaller utility loss.

---

### Q5. What does a NDCG of 0.1745 actually mean? Is that good?
**A:** For top-10 recommendation on a sparse implicit-feedback dataset with 10,362 items, 0.17 is a reasonable baseline — it's not a product system trained on billions of interactions. The absolute number matters less than the *relative change* across mitigations. A 1.7% drop in NDCG for a 20+ percentage point improvement in male exposure share is a very favorable tradeoff.

---

### Q6. How did you handle artists where MusicBrainz had no gender label?
**A:** Artists labeled "unknown" (bands, orchestras, unlabeled individuals) are excluded from all fairness metric calculations. They still appear in recommendations and affect utility metrics — we just don't count them toward exposure male share, rank gap, or coverage gap. This is a limitation: 62.7% of the catalog is excluded from fairness measurement.

---

### Q7. Isn't using binary male/female gender labels problematic?
**A:** Yes — this is explicitly acknowledged as our primary limitation. MusicBrainz only records binary gender for person artists. Non-binary, agender, and gender-fluid artists are either misclassified or labeled unknown. Our framework measures one dimension of a multi-dimensional fairness problem. A more inclusive approach would require richer metadata sources or self-reported artist data.

---

### Q8. Could the score boost harm female artists who are already popular (i.e., don't need the boost)?
**A:** The boost is applied uniformly to all female-labeled artists. In practice, already-popular female artists will tend to appear in recommendations regardless since their base ALS scores are high. The boost primarily helps lesser-known female artists break into recommendations for users who haven't heard them. A more targeted approach (boost only under-exposed artists) is a future direction.

---

### Q9. How did you define "relevant" items for Precision and Recall?
**A:** Relevant items are the artists a user actually interacted with in the held-out 20% test split. If ALS recommends an artist that the user listened to in the test period, that counts as a hit. This is standard evaluation for implicit-feedback recommenders.

---

### Q10. Why use position-weighted exposure instead of simple count of female artists recommended?
**A:** Position matters — an artist recommended at rank 1 gets far more clicks/streams than one at rank 10. Position-weighting (using 1/log₂(rank+1)) reflects real-world exposure more accurately. A simple count would treat a female artist at rank 10 the same as one at rank 1, underestimating how much rank affects actual visibility.

---

### Q11. What is the sensitivity sweep telling us?
**A:** It shows the *shape* of the fairness–utility tradeoff curve as we increase the female boost from 0 to 0.45. The curve is nearly flat in NDCG until about boost = 0.20 (Mit B = 0.28), then starts dropping more steeply. This tells us there is a "free zone" where significant fairness gains come at very low utility cost — the curve bends sharply around parity. This helps practitioners choose a boost value appropriate for their tolerance.

---

### Q12. Why Last.fm 360K? Is it a representative dataset?
**A:** Last.fm 360K is a standard benchmark in music recommendation research, available via HuggingFace, and contains real user listening histories with artist MusicBrainz IDs — essential for gender labeling. It skews toward Western music listeners and doesn't represent global listening patterns. Results may not generalize to streaming platforms with broader demographics.

---

### Q13. What would you do differently if you had more time?
**A:** 
1. **In-processing fairness:** modify the ALS loss function to penalize disparate exposure during training.
2. **Better label coverage:** use additional sources (Wikidata, Discogs) to label more of the 62.7% unlabeled artists.
3. **Non-binary gender:** work with datasets that include richer identity metadata.
4. **User-side analysis:** check if the recommendations are equally good for users of different demographics.
5. **Longitudinal simulation:** model how repeated re-ranking affects exposure over many recommendation rounds.

---

### Q14. How did you ensure the train/test split doesn't leak information?
**A:** We split per-user *before* building the ALS interaction matrix. Only training interactions go into the CSR matrix used to fit ALS. Test interactions are held out entirely and only used at evaluation time. The `filter_already_liked_items=True` flag in ALS's `recommend()` function ensures already-seen training items are excluded from recommendations.

---

### Q15. The coverage gap for Mit C is only 21 — why is that not ideal even though it's the most "equal"?
**A:** A coverage gap of 21 means roughly equal numbers of distinct male and female artists appear across all recommendations — which sounds fair. But the 37% NDCG collapse means those recommendations are largely irrelevant to users. Forcing parity through hard constraints sacrifices personalization. A fair system should ideally surface female artists that users are genuinely likely to enjoy, not just insert them arbitrarily.

---

*Prepared for DSCI 531 in-class presentation Q&A — April 2026*
