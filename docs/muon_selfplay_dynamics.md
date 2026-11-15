# How Muon’s Accelerated Learning Elevates Self-Play Game Quality

> **Note:** This technical note summarizes the empirical observations and optimization dynamics from our 9x9 Go self-play experiments using the Sayuri/KataGo hybrid pipeline. Drafted and edited in collaboration with Google AI.

---

In AlphaZero-style self-play reinforcement learning, the primary reason the **Muon optimizer** (particularly with Polar Express orthogonalization) dramatically elevates the quality of generated self-play games lies in its **spectral orthogonalization of gradient matrices**. By eliminating gradient anisotropy, Muon prevents rare, deep tactical discoveries from being drowned out by frequent positional patterns, thereby maximizing effective learning progress per training epoch.

This mechanism operates through a four-stage positive feedback loop:

## 1. Gradient Orthogonalization Accelerates the Absorption of Rare Tactics
Standard optimizers like SGD and Adam suffer from severe anisotropy in the singular value distribution of weight updates. Consequently, frequent, generic patterns (e.g., standard opening shapes) dominate the parameter updates, while **rare, high-leverage tactical signals—such as sharp vital points, complex ko threats, or one-liberty capturing races—receive negligible updates**.

* **Muon’s Mechanism:** Muon applies Newton-Schulz iterations (or faster-converging Polar Express coefficients) to the momentum matrix, normalizing all singular values toward `1`. This ensures that rare tactical directions receive the **same update step magnitude (spectral norm)** as common positional features.
* **Impact on Self-Play:** A deep tactical refutation discovered by Monte Carlo Tree Search (MCTS) in one epoch is rapidly internalized into the network’s **prior policy** by the next epoch, immediately elevating the baseline quality of subsequent self-play games.

## 2. High Learning Rates Enable a Tight "Search-to-Intuition" Loop
With traditional optimizers, using aggressive trunk learning rates causes dominant gradient directions to overshoot, leading to weight divergence or collapse. Because Muon strictly bounds the maximum singular value of the update matrix, it safely supports substantially higher learning rates.

* **Boosting MCTS Efficiency:** Under a fixed self-play playout budget (e.g., Gumbel AlphaZero with a few hundred visits), search depth depends heavily on the accuracy of the network's prior policy. Because Muon sharpens the policy head rapidly, MCTS avoids wasting visits on poor branches, achieving **effective reading depth equivalent to a much larger playout budget**.
* **The Virtuous Cycle:** Faster optimization produces sharper policy/value priors → sharper priors allow MCTS to uncover deeper, more accurate variations within the same playout limit → higher-purity training targets are fed back into the replay buffer.

## 3. Synergy with Surprise-Based (KLD) Priority Sampling
Modern self-play pipelines often employ **Policy Surprise Sampling** (weighting samples by the Kullback-Leibler divergence between the raw network policy and the post-search MCTS policy) to prioritize positions where search overturned the network’s intuition.

* **Targeted Blind-Spot Correction:** KLD sampling isolates the network’s exact tactical blind spots, and Muon’s orthogonalized updates overwrite those blind spots cleanly in fewer steps without destabilizing surrounding weights. As a result, basic reading oversights and endgame inaccuracies are purged from the self-play buffer far earlier in training.

## 4. Preserving Foundational Knowledge via Weight Decay Floors
In self-play, as the agent grows stronger, the distribution of game states shifts from chaotic midgames to razor-thin, half-point endgames. Aggressive learning rates risk causing catastrophic forgetting of basic life-and-death shapes learned earlier.

* **Orthogonal Expansion:** By pairing Muon with a **Weight Decay Floor** (halting decay once a channel's norm drops below a set fraction of its initialization RMS), the network retains its foundational feature extractors while accumulating fine-grained endgame and ko-fight representations in orthogonal subspaces.

---

## Summary
Where conventional optimizers require many small, noisy steps to gradually shift a model's strategic baseline, **Muon updates the network's intuition cleanly in a fraction of the steps**. Because self-play relies on the current model to generate its own future training data, this acceleration compounds across epochs—allowing compact models to reach professional-level tactical precision with significantly fewer self-play games.

---

[← Back to Main Technical Report](../README.md)
