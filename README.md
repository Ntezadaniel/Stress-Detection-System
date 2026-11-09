# Stress Detection via Knowledge Distillation — Project Scaffold

This is Step 1 of your thesis project: a working environment and data
pipeline you can run today, using synthetic data, before your real
WESAD download finishes.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1: Confirm your pipeline runs (do this now, no download needed)

```bash
python3 src/explore.py --synthetic
```

This generates fake wrist-sensor data with the same shape and sampling
rates as real WESAD data, then saves a plot to `outputs/signal_overview.png`.
Open that plot. You should see:
- BVP, EDA, TEMP, ACC signals with plausible shapes
- A label track at the bottom clearly showing baseline → stress →
  baseline → amusement → baseline blocks

If that plot looks right, your code is correct. The synthetic data itself
means nothing — it exists only to prove the pipeline works.

## Step 2: Get the real dataset

1. Download WESAD (2.5GB) from:
   https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/
2. Extract it so you end up with:
   ```
   data/raw/WESAD/S2/S2.pkl
   data/raw/WESAD/S3/S3.pkl
   ...
   data/raw/WESAD/S17/S17.pkl
   ```
   (WESAD skips S1 and S12, so don't worry if those are missing.)
3. Run the same exploration script on a real subject:
   ```bash
   python3 src/explore.py --subject S2
   ```
4. Compare this plot to your synthetic one. Real EDA and BVP signals should
   visibly change shape between the baseline and stress blocks — that
   visible difference is the signal your model has to learn to detect.
   If you can't see any difference by eye, that's worth investigating
   before you write a single line of model code.

## What's in this scaffold

```
src/
  data_loader.py      — loads real WESAD .pkl files, subject by subject
  synthetic_data.py   — generates fake data with matching shapes, for
                         testing your pipeline before real data is ready
  explore.py          — Step 1 script: load one subject and visualize it
  student_models.py   — defines several student model sizes (tiny/small/
                         medium/target) plus a matching teacher, for the
                         comparison in Step 5 and Step 7
  losses.py           — plain distillation loss, and the reinforced
                         (cost-sensitive) version with a penalty weight
                         for missed high-stress samples, for Step 5/6
requirements.txt       — Python dependencies
outputs/                — generated plots land here
```

## The road ahead (do not build these yet — one step at a time)

1. **Data pipeline** (this scaffold) — load and visualize raw signals.
2. **Preprocessing** — segment signals into fixed windows (e.g. 60s,
   50% overlap), align the different sampling rates, normalize.
3. **Classical baseline** — a Random Forest on hand-crafted features,
   as a fast sanity check before deep learning.
4. **Teacher model** — train a larger model (e.g. an LSTM or small
   transformer) on the windowed features to classify baseline/stress/
   amusement. Get this working and measure its accuracy first, before
   touching distillation at all.
5. **Distill several student sizes** — see `student_models.py` for
   four candidate sizes, including one (`student_target`) sized to
   land near your proposal's ~74% compression target. Train each
   against the teacher's soft outputs using `losses.py`'s plain
   `distillation_loss()` first.
6. **Add the cost-sensitive penalty** — once a plain student trains
   successfully, retrain your best candidate with
   `reinforced_distillation_loss()`, which weights high-stress samples
   `alpha` times more heavily so the model learns to avoid missing them.
7. **Evaluation** — compare teacher vs. every student candidate on
   accuracy, size, latency, and (for the reinforced version) false-
   negative rate on high-stress samples.
8. **Edge deployment simulation** — measure or estimate inference
   latency on constrained hardware.

Come back once Step 1 is confirmed working on real WESAD data, and we'll
build Step 2 (the preprocessing/windowing pipeline) together.
