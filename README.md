Repo forked from 

[Temporal Difference Learning for Model Predictive Control](https://arxiv.org/abs/2203.04955) by

[Nicklas Hansen](https://nicklashansen.github.io), [Xiaolong Wang](https://xiaolonw.github.io)\*, [Hao Su](https://cseweb.ucsd.edu/~haosu)\*
---
# TD-MPC Plasticity Instrumentation

This repository is a **fork of the original TD-MPC implementation** [https://github.com/nicklashansen/tdmpc?tab=readme-ov-file], extended with tools to analyze **plasticity in deep reinforcement learning models**.

It serves as the experimental baseline used to study how architectural modifications affect the learning dynamics of TD-MPC.

---

# Purpose

The goal of this repository is to provide:

1. A **baseline TD-MPC implementation**
2. Incremental architectural modifications
3. Detailed **plasticity diagnostics during training**

This setup enables controlled experiments comparing:

- baseline TD-MPC
- TD-MPC + SimNorm
- TD-MPC + SimNorm + LayerNorm
- TD-MPC + SimNorm + LayerNorm + Mish
- 
# Plasticity Metrics

The repository logs several metrics during training:

| Metric | Purpose |
|------|------|
| Weight magnitude | Measures overall parameter growth |
| Weight distance | Measures parameter drift |
| Gradient norm | Detects gradient explosion or collapse |
| Zero Gradient Ratio | Detects inactive gradients |
| Feature Zero Activation Ratio | Measures sparsity of learned representations |
| Stable rank | Estimates effective feature dimensionality |
| ENTK | Captures representation dynamics |


