markdown
# Experiment 1: Baseline `b6xc96h32` Run (512K Steps)

## Overview
Our initial benchmark established the baseline strength of a 6-block, 96-channel hybrid CNN–Transformer network (`R-R-T-R-R-T`) trained from scratch using standard Polar Express Muon and Gumbel AlphaZero on 9x9 Go.

## Configuration
* **Topology:** `b6xc96h32` (`96` channels, `3` heads × `32` dim, `SwiGLU`, `Fixup`, `RepLK` policy head)
* **Optimization:** Standard Muon (`ns_steps=5`, `WdFloorFraction=0.5`) + AuxAdamW
* **Training Horizon:** `512,000` steps (`131M` samples)

## Results vs. `SAI (b8c160)` (200 Games)
* **Black (Komi 7.0):** `17W – 20D – 63L` (37.0% non-loss)
* **White (Komi 7.0):** `14W – 17D – 69L` (31.0% non-loss)
* **Total:** **`31W – 37D – 132L` (`34.0%` non-loss rate)**

## Key Observations
* Demonstrated that a compact 6-block hybrid architecture trained purely via self-play could compete against an 8-block `160ch` reference engine.
* Revealed bottlenecks in late-game ko resolution and half-point endgame precision (`scores loss ~0.11`), motivating the search curriculum and width-scaling experiments in Exp 2 and Exp 3.

[← Back to Main Report](../README.md)
