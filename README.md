# Amplified Silence — post-processing fairness pipeline

Music recommendation (implicit ALS on Last.fm-style listening data) with **post-hoc** fairness interventions: score boosting and constrained re-ranking for artist gender exposure, evaluated with ranking utility (Precision / Recall / NDCG) and provider-side exposure metrics.

**Repository:** [https://github.com/DHYEY166/Amplified-Silence](https://github.com/DHYEY166/Amplified-Silence)

This repo contains **code and documentation only**. The course report itself is **not** included here (it is submitted separately to the instructor).

## Code documentation

The pipeline is split into small modules with **module docstrings** and **comments** where logic is non-obvious (e.g. train/test construction, fairness aggregation, mitigation scoring). Start from `run_experiment.py` (CLI entry point and experiment flow), then `src/data.py`, `src/labels.py`, and `src/evaluate.py`.

## Reproducing results (for reviewers)

Follow these steps **in order** from a fresh clone so dependencies and outputs line up.

### 0. Prerequisites

- **Python 3.10+** (tested through 3.12).
- **Network access** on the first run: the Hugging Face dataset is downloaded on demand; **MusicBrainz** is queried unless you use `--skip-mb-fetch` with an existing label cache.
- A writable directory for the Hugging Face cache (often a few hundred MB over time). For **[this GitHub repository](https://github.com/DHYEY166/Amplified-Silence)**, set `HF_HOME` inside the clone (step 3). If you keep a **local copy** of this code under a nested `post_processing/` folder inside some other project, you can point `HF_HOME` to a cache beside that parent tree instead.

### 1. Repository root

After cloning, work in the folder that contains **`run_experiment.py`** (the **repository root** for [Amplified-Silence](https://github.com/DHYEY166/Amplified-Silence)):

```bash
git clone https://github.com/DHYEY166/Amplified-Silence.git
cd Amplified-Silence
```

If you use a **local monorepo** and this code lives in a nested `post_processing/` directory, `cd` there instead.

All commands below assume your shell’s current directory is that folder (the one with `run_experiment.py`).

### 2. Create an environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `pip install implicit` fails (uncommon on current macOS/Linux wheels), install a C/C++ toolchain for your platform and retry, or use a conda environment with `implicit` from conda-forge.

### 3. (Optional) Hugging Face token and cache

Not required for public data, but can improve download rate limits:

```bash
export HF_TOKEN="your_token"
```

**Recommended** for a standalone clone of this repo (keeps the cache inside the project):

```bash
export HF_HOME="$PWD/.hf_cache"
export HF_DATASETS_CACHE="$HF_HOME"
```

**Optional:** if this project lives as `…/post_processing/` inside a larger parent project and you want a shared cache next to that parent:

```bash
export HF_HOME="$PWD/../.hf_cache"
export HF_DATASETS_CACHE="$HF_HOME"
```

### 4. Quick smoke test (verifies the pipeline runs)

Uses a small streaming slice and a **small** MusicBrainz budget so a reviewer can confirm the stack in minutes (not hours):

```bash
python run_experiment.py --streaming --split 'train[:250000]' \
  --min-user 15 --min-artist 8 \
  --max-mb-fetch 150 \
  --factors 32 --als-iters 12 --k 10
```

**Success:** `data/outputs/results.json` exists and the script exits with code 0. Metrics will differ from the paper’s full-scale run.

### 5. Full experiment (closer to paper-scale settings)

Requires more time (dataset download + many MusicBrainz calls + ALS):

```bash
python run_experiment.py --streaming --split 'train[:1200000]' \
  --min-user 25 --min-artist 15 \
  --max-mb-fetch 3000 \
  --factors 64 --als-iters 20
```

To **re-run without hitting MusicBrainz** after you already have `data/cache/musicbrainz_artist_labels.json`:

```bash
python run_experiment.py --streaming --split 'train[:1200000]' \
  --min-user 25 --min-artist 15 \
  --skip-mb-fetch \
  --factors 64 --als-iters 20
```

### 6. Figures (after step 4 or 5)

```bash
python generate_figures.py
```

Creates `data/outputs/figures/` (`fig1_*.png` through `fig4_*.png` when `results.json` is present).

### 7. Presentation (optional)

`make_presentation.py` expects the **official USC PowerPoint template** file  
`USC_PP_Template_General_National2_16x9.pptx` in the **repository root** (same folder as `run_experiment.py`). That template is **not** included in this repository (USC brand restrictions / local distribution only). Obtain it from your USC unit or [USC Communications](https://brand.usc.edu/) and place it there with that exact filename.

Also requires figures under `data/outputs/figures/` (from step 6):

```bash
python make_presentation.py
```

This writes `AmplifiedSilence_Presentation.pptx` in the repository root.

---

## Data and terms of use

- Interactions are loaded from the Hugging Face mirror [`matthewfranglen/lastfm-360k`](https://huggingface.co/datasets/matthewfranglen/lastfm-360k), derived from the **Last.fm 360K** dataset.
- Last.fm’s data is **for non-commercial use**; see the dataset card and [Last.fm API terms](https://www.last.fm/api/tos) if you redistribute or build on this project.

Artist gender / type labels are fetched from **MusicBrainz** (see `src/labels.py`). The client uses a descriptive `User-Agent` and sleeps between requests.

## CLI reference (`run_experiment.py`)

| Flag | Meaning |
|------|--------|
| `--streaming` | Stream the HF split with `--split train[:N]` (recommended). |
| `--split` | Dataset slice, e.g. `train[:1200000]`. |
| `--skip-mb-fetch` | Only use existing `data/cache/musicbrainz_artist_labels.json`. |
| `--max-mb-fetch` | Cap new MusicBrainz lookups (incremental cache). |
| `--out-dir` | Output directory (default: `data/outputs`). |

Generated artifacts (default locations):

- `data/outputs/results.json` — baseline, mitigations A/B/C, sensitivity sweep.
- `data/outputs/tradeoff_*.png` — exposure and NDCG vs boost.

**Note:** `data/cache/`, `data/outputs/`, and `data/runs/` are typically **gitignored**; after cloning, run the experiment locally to create them.

## Project layout

| Path | Role |
|------|------|
| `run_experiment.py` | End-to-end: data → ALS → evaluation → mitigations |
| `src/data.py` | HF load, filter, train/test split, sparse matrix |
| `src/labels.py` | MusicBrainz label cache |
| `src/evaluate.py` | NDCG / precision / recall + fairness accumulators |
| `generate_figures.py` | Paper-style figures from `results.json` |
| `make_presentation.py` | Build deck from USC template |

## Citation

If you use the original Last.fm 360K dataset, cite Celma (2010) / the dataset page you obtained it from, and comply with Last.fm’s non-commercial conditions.
