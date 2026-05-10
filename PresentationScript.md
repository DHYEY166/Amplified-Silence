# Presentation Script — Amplified Silence (DSCI 531)
**Total time: ~8 minutes | Q&A: 2 minutes**

---

## Slide 1 — Title (~ 20 seconds)

Hi everyone. Our project is called **Amplified Silence** — the idea being that female artists are being systematically silenced not by explicit decisions, but by the quiet amplification of bias inside recommendation algorithms. Today we'll walk you through the problem, what we built, and what we found. I'm Dhyey, and I'm joined by Shivam and Tejas.

---

## Slide 2 — Problem Description (~1 min 30 sec)

So, what's the problem?

Music streaming platforms don't just play songs — they **surface artists**. Who gets recommended directly affects whose career grows. And the data shows that female and non-binary artists consistently receive less algorithmic visibility than male artists.

This is a **provider-side fairness** problem. Most fairness research focuses on users — are recommendations equally good for everyone? We flip that and ask: are artists being treated fairly by the algorithm?

The feedback loop here is dangerous. An artist who gets recommended more gets more plays, which makes the algorithm recommend them even more. So even a small initial skew compounds over time.

For our dataset, we used **Last.fm 360K** — a real-world listening history dataset with 24,509 users and 10,362 artists and over a million play events. To measure gender bias, we fetched artist gender labels from MusicBrainz — an open music encyclopedia — through 11,000 API calls, getting labels for 3,867 artists. And right away, we found that 72.8% of all weighted listening in the training data goes to male artists. That's our baseline bias before any recommendation even happens.

---

## Slide 3 — Methodology: Data Pipeline & Model (~1 min 30 sec)

Let me walk through how we built this.

We pulled the Last.fm dataset from HuggingFace, filtered to users with at least 25 interactions and artists with at least 15 — to remove noise. We did an 80/20 per-user train-test split and used log-scaled play counts as confidence weights for the model.

For the recommendation model, we used **ALS — Alternating Least Squares** — which is the standard algorithm for implicit feedback. It learns a vector for each user and each artist in a shared latent space, where dot products predict affinity. We set 64 latent factors and ran 20 iterations.

One important design choice: instead of running the model separately for every fairness experiment, we precomputed a candidate pool of the top 400 artists per user once, then applied all our fairness re-rankings within that pool. This made the sensitivity sweep computationally feasible.

And we already observed the core problem here: 72.8% of training interactions go to male artists. This upstream skew gets embedded directly into the model's latent factors.

---

## Slide 4 — Methodology: Fairness Metrics & Mitigations (~1 min 30 sec)

We measured provider-side fairness using three metrics:

First, **exposure male share** — the fraction of position-weighted exposure going to male artists across all top-10 lists. Items ranked higher get more weight. This is our primary fairness metric.

Second, **average rank gap** — mean rank of male artists minus mean rank of female artists. Positive means males rank higher on average.

Third, **coverage gap** — how many more distinct male artists get recommended compared to female artists.

For mitigations — and importantly, all three require **no model retraining** — we tried:

**Mitigation A** multiplies every female artist's score by 1.12 before re-ranking. A gentle nudge.

**Mitigation B** uses a stronger multiplier of 1.28 — a more deliberate push toward parity.

**Mitigation C** is a hard constraint: force at least 4 female artists into every top-10 list, regardless of score.

And we also ran a sensitivity sweep — varying the boost from 0 to 0.45 — to understand the shape of the fairness-utility tradeoff, which brings us to our results.

---

## Slide 5 — Results: Exposure & Amplification (~1 min)

Looking at the left figure — you can see the exposure male share across all four models. The baseline sits at 69.8%, close to the 72.8% in training data, confirming ALS largely inherits the bias. Mit A brings it to 60.3%. Mit B gets to 49.2% — essentially parity. And Mit C drops all the way to 14.6%, which is an overcorrection.

The right figure shows the same story as a pipeline — how male exposure share changes from training data through each mitigation stage. The bias starts in the data, survives through ALS, and our post-processing corrections progressively bring it down toward parity.

---

## Slide 6 — Results: Fairness–Utility Tradeoff (~1 min)

Now the key question is — what does this cost in recommendation quality?

The curve on the left is our sensitivity sweep. Each dot is a different boost value. Moving left means fairer — less male dominance. The Y-axis is NDCG@10 — recommendation quality. 

What's striking is how **flat the curve is** from Baseline all the way to Mit B. You go from 69.8% male exposure down to 49.2% — a 20 percentage point fairness improvement — while NDCG drops by less than 2%. That's the "free zone."

Beyond that, you can see the curve bending steeply — each additional fairness gain costs significantly more utility. That's where Mit C lives — 14.6% male exposure but a 37% NDCG collapse.

The table on the right confirms this. Mit B is the clear sweet spot — near parity with essentially the same precision and recall as the baseline.

---

## Slide 7 — Takeaways & Limitations (~40 seconds)

To summarize our key findings:

ALS amplifies the training data skew. Score boosting is an effective and cheap fix — no retraining needed. And Mitigation B, a +28% female score boost, hits near-parity at under 2% utility cost.

That said, we have real limitations. We're working with binary gender labels, which excludes non-binary identities entirely. We only have labels for 37% of the catalog. And our mitigations are post-processing only — the upstream bias in training data is still there. Future work would look at in-processing fairness constraints and richer identity metadata.

Thank you — happy to take questions.

---

*Tip: Speak slowly, make eye contact, and pause briefly after each figure before moving on. Aim for ~8 minutes total.*
