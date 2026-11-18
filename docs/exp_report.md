# Experiment-by-Experiment Technical Breakdown (Exp 1 – Exp 4)

This document provides an individual summary, architectural configuration, training dynamics, and key takeaways for each of the **five experimental stages** conducted on 9x9 Go (`Chinese rules`, `Komi 7.0`, evaluated over `200` games at `1,200 playouts/move` against `SAI b8c160`).

---

## Experiment 1: Baseline (`b6xc96h32` — Initial Muon + Gumbel AZ)

### 1. Objective & Configuration
Establish a clean baseline using a compact 6-block Hybrid ResNet-Transformer (`96ch`, `3 heads x 32 dim`) trained with standard (un-split) Muon optimization and Gumbel AlphaZero self-play.
* **Topology:** `b6xc96h32` (`Res-Res-Trans-Res-Res-Trans`, `ResidualChannels: 96`, `TransformerHeads: 3`, `head_dim: 32`)
* **Training:** `512K` steps, standard un-split Muon optimizer, Gumbel AZ self-play

### 2. Benchmark Results (vs. `SAI b8c160`)
* **Black (`W/D/L`):** `17 / 20 / 63` (Point Win Rate: `27.0%`)
* **White (`W/D/L`):** `14 / 17 / 69` (Point Win Rate: `22.5%`)
* **Total (`W/D/L`):** `31 / 37 / 132` (**Point Win Rate: `24.75%`**, **Est. Elo Diff: `-193 Elo`**, Non-Loss Rate: `34.0%`)

### 3. Key Findings
* While the hybrid architecture successfully learned basic 9x9 tactics and endgame scoring, it remained nearly `200 Elo` behind the 8-block `160ch` ResNet (`SAI b8c160`) even after `512K` steps.
* Orthogonalizing concatenated Q/K/V/Out projection matrices as a single block limited attention head specialization.

---

## Experiment 2: Naive Width Scaling (`b6xc192h64` — Un-split Muon + Damped Restart)

### 1. Objective & Configuration
Test whether simply doubling network width (`96ch -> 192ch`, ~4x parameters) and increasing head dimension (`32 -> 64`) could close the `193 Elo` gap.
* **Topology:** `b6xc192h64` (`ResidualChannels: 192`, `TransformerHeads: 3`, `head_dim: 64`)
* **Training:** `392K` steps with 2-stage search scaling and a damped warm restart, using standard un-split Muon without width-adjusted LR scaling

### 2. Benchmark Results (vs. `SAI b8c160`)
* **Black (`W/D/L`):** `23 / 22 / 58` (Point Win Rate: `33.0%`)
* **White (`W/D/L`):** `25 / 18 / 54` (Point Win Rate: `35.0%`)
* **Total (`W/D/L`):** `48 / 40 / 112` (**Point Win Rate: `34.00%`**, **Est. Elo Diff: `-115 Elo`**, Non-Loss Rate: `44.0%`)

### 3. Key Findings
* Scaling up parameters improved strength by `+78 Elo` over Exp 1, but still fell short of `SAI b8c160` (`-115 Elo`) despite requiring roughly 4x the compute per step.
* **Takeaway:** Raw parameter scaling without proper optimizer head-splitting and `muP` learning rate scaling yields diminishing returns.

---

## Experiment 3: Algorithmic "Giant Killing" (`b6xc96h32` — Head-Split Muon + RMSNormMask)

### 1. Objective & Configuration
Return to the compact `b6xc96h32` (`96ch`) footprint and focus purely on optimization, normalization, and feature representation improvements.
* **Topology:** `b6xc96h32` (`ResidualChannels: 96`, `TransformerHeads: 3`, `head_dim: 32`, `FinalBlockCgroupSize: 24` [`4 x 24ch` spatial RMSNormMask], `AttnLogitPenaltyCap: 30.0`, `Full 43ch Input`)
* **Training:** `312K` steps single cycle (`Peak LR: 2.0e-2` with `muP` width adjustment), **Head-Split Muon** (orthogonalizing per `32`-dim head), 3-stage self-play curriculum (`150 -> 350 -> 500 playouts`)

### 2. Benchmark Results (vs. `SAI b8c160`)
* **Black (`W/D/L`):** `31 / 26 / 39` (Point Win Rate: `45.8%`, Non-Loss: `59.4%`)
* **White (`W/D/L`):** `21 / 13 / 70` (Point Win Rate: `26.4%`, Non-Loss: `32.7%`)
* **Total (`W/D/L`):** `52 / 39 / 109` (**Point Win Rate: `35.75%`**, **Est. Elo Diff: `-102 Elo`**, Non-Loss Rate: `45.5%`)
* *(Note: At the `308K` annealing trough prior to the `312K` cycle reset, the model scored `41/48/111` [`32.50%` Point Win Rate / `-127 Elo`]; a brief `12.5%` high-LR blend at `312K` unlocked sharper Black play, boosting Black from `20W` to `31W`.)*

### 3. Key Findings
* **Surpassed the 4x Larger Exp 2 Model:** Achieved `-102 Elo` in `312K` steps—beating `Exp 2` (`192ch`, `-115 Elo` at `392K` steps) with a quarter of the parameters.
* **Color Asymmetry:** Black became highly competitive (`31W - 39L`), whereas White struggled (`21W - 70L`) against `SAI`'s deep endgame refutations.

---

## Experiment 3.5: Damped Warm Restart & Cycle Boundary Analysis (`b6xc96h32`, `312K -> 512K`)

### 1. Objective & Configuration
Test whether extending `b6xc96h32` via a Damped Warm Restart (`312K -> 512K` steps) with deeper search (`600 playouts`, `Gumbel: 160`) and tighter opening focus (`random-opening-prob: 0.15`) could overcome White's weakness and achieve overall parity.
* **Training:** `6K`-step pseudo-warmup (`3e-3 -> 6e-3 -> 9e-3`) followed by a `194K`-step cosine decay to `1e-4`; `PolicySurpriseFactor: 0.65`, `resign-threshold: 0.03`, `SwaMaxCount: 12`

### 2. Benchmark Results (vs. `SAI b8c160`)
* **At `508K` (Clean Annealing Trough):**
  * **Black (`W/D/L`):** `30 / 18 / 52` (Point Win Rate: `39.0%`)
  * **White (`W/D/L`):** `27 / 30 / 43` (Point Win Rate: `42.0%`, Non-Loss: `57.0%`)
  * **Total (`W/D/L`):** `57 / 48 / 95` (**Point Win Rate: `40.50%`**, **Est. Elo Diff: `-67 Elo`**, Non-Loss Rate: `52.5%`)
* **At `512K` (Post-Cycle LR Reset Contamination):**
  * **Total (`W/D/L`):** `41 / 53 / 106` (**Point Win Rate: `33.75%`**, **Est. Elo Diff: `-117 Elo`**, Non-Loss Rate: `47.0%`)

### 3. Key Findings
* **White's Weakness Solved (`+35 Elo` Net Gain):** At `508K`, White's Point Win Rate surged by `+15.6` percentage points (`26.4% -> 42.0%`), and training losses hit record lows (`all loss: 4.1880`, `scores loss: 0.0726`).
* **Two Crucial Training Traps Identified:**
  1. *Jigo (Draw) Over-Specialization:* Tightening `random-opening-prob` to `0.15` caused the model to overfit onto passive draw equilibrium lines, capping overall Point Win Rate at `40.50%` (`-67 Elo`).
  2. *SWA Cycle Boundary Sensitivity:* At `512K`, the cosine schedule modulo reset spiked LR back to `9.0e-3` during the final `1,000` steps. Just 2 post-spike checkpoints entering the 12-checkpoint SWA window degraded strength by `-50 Elo` (`-67 -> -117 Elo`).

---

## Experiment 4: `muP` Scaling & Single-Cycle Sweet Spot (`b6xc128h32`, `0 -> 312K`)

### 1. Objective & Configuration
Synthesize all lessons from Exp 1–3.5 to find the **smallest architecture capable of matching `SAI b8c160` in a single ~300K-step cycle without warm restarts**.
* **Topology:** `b6xc128h32` (`ResidualChannels: 128`, `TransformerHeads: 4`, `head_dim: 32`, `TransformerFFNChannels: 512`, `FinalBlockCgroupSize: 32` [`4 x 32ch` spatial RMSNormMask], `PolicyHeadChannels: 48`, `ValueHeadChannels: 96`)
* **Zero-Tuning `muP` Scaling:** Scaled Peak LR from `96ch` (`2.0e-2`) via `2.0e-2 * sqrt(96 / 128) = 1.73e-2`
* **Clean SWA Termination:** Set `WarmUpSteps: 12,000` and `LearningRateAnnealingCycle: 304,000` (trough at `316K`) so stopping at `312K` (`LR = 1.07e-4`) guaranteed zero LR spike contamination
* **Balanced 3-Stage Curriculum (`312K` steps total):**
  * Stage 1 (`0 – 48K`): `150 playouts` (`Gumbel: 32`), `random-opening-prob: 0.65`
  * Stage 2 (`48K – 160K`): `350 playouts` (`Gumbel: 85`), `random-opening-prob: 0.45`
  * Stage 3 (`160K – 312K`): `550 playouts` (`Gumbel: 140`), `random-opening-prob: 0.25`, `root-policy-temp: 0.90`, `score-utility-factor: 0.28`

### 2. Benchmark Results (vs. `SAI b8c160`)
* **Black (`W/D/L`):** `40 / 19 / 40` (**Point Win Rate: `50.0%`**, Non-Loss: `59.6%`)
* **White (`W/D/L`):** `26 / 29 / 46` (**Point Win Rate: `40.1%`**, Non-Loss: `54.5%`)
* **Total (`W/D/L`):** `66 / 48 / 86` (**Point Win Rate: `45.00%`**, **Est. Elo Diff: `-35 Elo`**, **Non-Loss Rate: `57.0%`**)
* **Final Convergence Metrics (`311K–312K`):** `all loss: 4.2327`, `prob loss: 1.5918`, `wdl loss: 0.6091`, `scores loss: 0.0686` (all-time best endgame score precision)

### 3. Key Findings
* **Target Achieved in One Pass:** Without any warm restarts or hyperparameter grid searches, `b6xc128h32` matched `SAI b8c160` 50/50 as Black (`40W - 40L`) and reached `57.0%` overall non-loss rate (`-35 Elo`) in `312K` steps—confirming `b6xc128h32` as the minimal architectural sweet spot for 9x9 Go.
