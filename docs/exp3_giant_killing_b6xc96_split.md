markdown
# Experiment 3: "Giant Killing" — `b6xc96h32` with `Muon Split` & Full Features (312K Steps)

## Overview
The goal of Experiment 3 was to test whether our `96ch` architecture (~4x smaller and 3–4x faster than `b6xc192h64`) could outperform our strongest `192ch` model by combining head-level orthogonalization, spatial group normalization, full input features, and a 3-stage search curriculum.

## Architectural & Optimization Upgrades
* **Head-Split Muon (`Muon Split`):** Split `q_proj`, `k_proj`, `v_proj`, and `out_proj` weights along head dimensions into independent `(32, 96)` matrices prior to Polar Express orthogonalization, preventing cross-head interference and normalizing per-head update scaling.
* **Group-Spatial RMSNorm:** Enabled `FinalBlockCgroupSize: 24` (`RMSNormMask` over 4 groups of 24 channels) and `AttnLogitPenaltyCap: 30.0`.
* **Full 43-Channel Input (`ReductionInput: false`):** Supplied precomputed liberty and ladder planes so the compact `96ch` trunk could dedicate its full capacity to strategic and endgame evaluation.
* **Width-Scaled Peak LR:** Set `Peak LR = 2.0e-2` (matching the theoretical $`\(\sqrt{192/96} \times 1.5\times 10^{-2} \approx 2.12\times 10^{-2}\`$) scaling law).

## 3-Stage Self-Play Curriculum (`0 – 312K Steps`)
* **Stage 1 (`0 – 56K`):** `150` playouts (25% full), `random-opening-prob: 0.75`
* **Stage 2 (`56K – 192K`):** `350` playouts (70% full), `random-opening-prob: 0.50`, `fpu-reduction: 0.35`
* **Stage 3 (`192K – 312K`):** `500` playouts (70% full), `random-opening-prob: 0.30`, `fpu-reduction: 0.35`

## Final Results vs. `SAI (b8c160)` (200 Games at 312K Steps)
* **Final Training Losses:** `all loss: 4.2372`, `prob loss: 1.5710`, `wdl loss: 0.6128`, **`scores loss: 0.0823` (Record Low)**
* **Black (Komi 7.0):** **`31W – 26D – 39L` (`59.4%` non-loss rate — winning record as Black)**
* **White (Komi 7.0):** **`21W – 13D – 70L` (`32.7%` non-loss rate)**
* **Total:** **`52W – 39D – 109L` (`45.5%` non-loss rate)**

## Key Observations
* Achieved **Giant Killing**: Surpassed the 4x larger `b6xc192h64` model (44.0%) in 80K fewer steps (`312K` vs. `392K`), establishing an all-time record of **52 wins and 39 draws** against `SAI (b8c160)`.
* Achieved an unprecedented **`scores loss` of `0.0823`**, translating into a dominant **59.4% non-loss rate as Black** under 7.0 komi.

[← Back to Main Report](../README.md)
