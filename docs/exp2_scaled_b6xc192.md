markdown
# Experiment 2: Scaled `b6xc192h64` with Progressive Search & Damped Restart (392K Steps)

## Overview
To test capacity scaling and curriculum design, we doubled network width to `192` channels (`b6xc192h64`, ~4x parameters) while intentionally using a compact 27-channel input (`ReductionInput: true`, omitting precomputed liberty/ladder planes) to test end-to-end tactical feature learning.

## Configuration & Curriculum
* **Topology:** `b6xc192h64` (`192` channels, `3` heads × `64` dim, `ReductionInput: true`, `FinalBlockCgroupSize: 0`)
* **Wave 1 (`0 – 300K steps`):** `Peak LR: 1.5e-2 → 5e-4` (`12K` warmup, `300K` cycle). Self-play escalated at `88K` from `150` playouts (25% full) to `350` playouts (70% full).
* **Wave 2 (`300K – 392K steps`):** Damped cosine restart (`2.5e-3 → 2e-4`, `80K` cycle) paired with `500` playouts (70% full) and reduced opening randomness (`0.30`).

## Progression vs. `SAI (b8c160)` (200 Games per Checkpoint)

| Checkpoint | `all loss` | `prob loss` | `wdl loss` | `scores loss` | Black (`W/D/L`) | White (`W/D/L`) | Total (`W/D/L`) | Non-Loss Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **88K** | 4.8182 | 1.8777 | 0.5557 | 0.1821 | 0 / 3 / 95 | 0 / 0 / 102 | 0 / 3 / 197 | **1.5%** |
| **180K** | 4.5406 | 1.7035 | 0.6413 | 0.1206 | 9 / 8 / 69 | 9 / 12 / 93 | 18 / 20 / 162 | **19.0%** |
| **284K** | 4.0443 | 1.5038 | 0.5128 | 0.0915 | 24 / 14 / 61 | 19 / 23 / 59 | 43 / 37 / 120 | **40.0%** |
| **360K** | 4.1824 | 1.5722 | 0.5347 | 0.0946 | 24 / 20 / 48 | 21 / 21 / 66 | 45 / 41 / 114 | **43.0%** |
| **392K** | **4.1560** | **1.5740** | **0.5246** | **0.0889** | **23 / 22 / 58** | **25 / 18 / 54** | **48 / 40 / 112** | **44.0%** |

## Key Observations
* Surpassed the Exp 1 baseline by **180K steps** and reached a **44.0% non-loss rate** at 392K steps.
* Proved that a single damped cosine restart paired with higher self-play playouts (`350 → 500`) effectively breaks late-stage plateaus (`40.0% → 44.0%`).

[← Back to Main Report](../README.md)
