# Uplift Modeling for Promotion Allocation

A causal ML demo built while applying for the **Senior Machine Learning Scientist (Accommodations)**
role on Booking.com's ABU ML team (Margin Management, Accommodation Business Unit).

The team's own recent publication, *["Converted Data is All You Need for Causal Optimization of
e-Commerce Promotions"](https://arxiv.org/) (ACM CIKM 2025)*, and their earlier work on
budget-constrained promotion personalization via uplift modeling, motivated the specific angle
of this demo: rather than a generic "I know sklearn" project, this builds a small, complete
pipeline addressing the JD's core technical asks directly —

- Heterogeneous treatment effect (uplift) estimation
- Observational debiasing (IPW, doubly robust estimation)
- Neural network architecture design for tabular data (attention)
- Offline evaluation methods that reliably predict online performance (Qini/AUUC vs. PEHE)

## The problem, briefly

A company can show a promo to a user. The goal isn't "who's most likely to book" — it's "whose
behavior does the promo actually change." Spending budget on users who'd book anyway wastes
margin; spending on users the promo never convinces wastes budget; and some users are actively
hurt by a promo (they'd have booked at full price, and now get an unnecessary discount). This
is a causal inference problem, not a plain prediction problem, because the historical decision
to show a promo was not random — it was already targeted, which biases naive comparisons.

## Honesty note on the data

**This demo uses synthetic data, not Booking.com data.** I don't have access to the company's
real data, and building a demo that implied otherwise would be misleading. The data-generating
process (`src/simulate.py`) is calibrated to be *qualitatively* realistic — confounded treatment
assignment, positivity/overlap, non-trivial and non-monotonic heterogeneous effects across four
uplift quadrants — rather than fit to Booking's actual figures. The point of using synthetic
data is that it comes with a known ground-truth treatment effect, which lets the pipeline be
validated against the truth (PEHE) in a way that's never possible with real-world data, where
only the factual outcome is ever observed. See `docs/interview_prep_causal_uplift_modeling.md`
for a full discussion of this trade-off, including how one would calibrate a DGP against real
data or published benchmarks if they were available.

## Results

Four models, in increasing order of sophistication, evaluated on a held-out test set:

| Model | PEHE (vs. ground truth, lower better) | AUUC (observed data only, higher better) |
|---|---|---|
| S-learner (naive) | 12.62 | 11,135 |
| T-learner (naive) | 11.52 | 11,587 |
| T-learner + IPW | 11.40 | 11,711 |
| **Attention-DragonNet** | **10.76** | **12,714** |

Both metrics — one only computable because ground truth is known (PEHE), one computable only
from observed data the way a real deployment would be (AUUC/Qini) — agree on the same ranking.
That agreement is the strongest evidence that debiasing and the attention architecture
genuinely improved targeting quality, not an artifact of a metric that only works in a lab
setting.

One honest caveat: the learned targeted-regularization epsilon in the DragonNet converged very
close to zero, meaning the explicit doubly-robust *correction term* contributed only a small
increment on this dataset — most of the improvement over the plain IPW T-learner likely comes
from the shared attention representation learning feature interactions (e.g. loyalty tier ×
price sensitivity) better than gradient-boosted trees or an MLP would. Worth having as a direct
answer if asked, rather than overselling the doubly-robust piece specifically.

## Architecture: Attention-DragonNet

```
loyalty_tier ──────► embedding ┐
device_mobile ──────► embedding ┤
price_sensitivity ──► linear    ├──► [CLS] + tokens ──► self-attention ──┬──► propensity head
lead_time_days ─────► linear    │        (2 layers,        (shared          ├──► y0 head (control)
past_bookings ──────► linear    │         4 heads)          repr.)          └──► y1 head (treated)
length_of_stay ─────► linear    │
base_price ─────────► linear    ┘
```

Each feature is tokenized into its own embedding, a self-attention trunk lets features attend
to each other (learning interactions like loyalty tier × price sensitivity directly, rather than
relying on a tree's implicit splits), and the pooled `[CLS]` representation feeds three heads —
propensity, and two potential-outcome heads — following Dragonnet (Shi, Blei & Veitch, 2019).
A learnable targeted-regularization term (`epsilon`) nudges the factual outcome prediction using
the inverse-propensity-weighted residual direction, giving a doubly-robust-style correction
trained end-to-end rather than applied as a separate post-hoc step. Full explanation, with
analogies, in `docs/interview_prep_causal_uplift_modeling.md`.

## Repo structure

```
├── src/
│   ├── simulate.py      # the data-generating process — confounded, heterogeneous
│   ├── evaluation.py    # PEHE + Qini curve / AUUC
│   └── model.py          # Attention-DragonNet (PyTorch)
├── notebooks/
│   ├── 01_data_and_confounding.ipynb    # the problem, EDA
│   ├── 02_naive_baseline.ipynb          # S-learner / T-learner baselines
│   ├── 03_attention_dragonnet.ipynb     # training the neural model + IPW T-learner
│   └── 04_results_and_evaluation.ipynb  # Qini curves, headline comparison
├── models/                # saved model artifacts (generated by notebooks 02-03)
├── data/                  # saved results tables + test set (generated by notebooks)
├── figures/               # all plots, baked into the notebooks
├── web/                    # HTML demo (Flask backend + static frontend)
│   ├── server.py            # API: /api/predict, /api/results, /api/qini
│   └── static/               # index.html, results.html, styles.css, app.js
├── docs/
│   └── interview_prep_causal_uplift_modeling.md   # concept reference doc
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# Run notebooks in order (each saves artifacts the next one needs)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_and_confounding.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_naive_baseline.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_attention_dragonnet.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_results_and_evaluation.ipynb

# Then launch the interactive demo (serves both screens + API on :5050)
python web/server.py
```

Open `http://localhost:5050`. Two screens: **customer profile**, where you build a synthetic
customer and see all four models' predicted uplift live (with a targeting-quadrant read), and
**full results** (`/results.html`), showing the Qini curves, metrics table, and notebook figures.
The frontend is plain HTML/CSS/JS with no build step or client-side framework — Flask serves the
static files and a small JSON API (`/api/predict`, `/api/results`, `/api/qini`) that reuses the
exact same saved model artifacts as the notebooks.

## What this demo does *not* cover

The JD also calls out marketplace interference/cannibalization (SUTVA violations — one
property's promo affecting a competing property's bookings) and production pipeline design
(Spark/Airflow at scale). Both are out of scope for a 2-day demo, but are discussed conceptually
in the reference doc, since they're a natural and important follow-up direction.
