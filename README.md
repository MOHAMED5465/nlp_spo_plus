# NLP-Driven Decision-Focused Learning for CVRP

This repository explores **Decision-Focused Learning (DFL)** for the Capacitated Vehicle Routing Problem (CVRP), combining Natural Language Processing (NLP), Machine Learning, and Combinatorial Optimization.

The objective is to learn routing costs from natural-language descriptions of operational conditions, such as traffic, weather, and local events, while accounting for the impact of prediction errors on downstream routing decisions.

Rather than optimizing prediction accuracy alone, the project investigates whether decision-focused training can improve routing quality, measured through **decision regret**, compared with conventional regression-based approaches.

## 1. Problem Formulation

We consider a CVRP defined on a directed graph \(G=(V,E)\), where edge costs depend on operational conditions described in textual inputs.

A neural network \(f_\theta\) predicts the vector of edge costs \(\hat{c}\) from these inputs. The predicted costs are then used to solve the routing problem:

\[
\hat{w} \in \arg\min_{w\in S_{\mathrm{CVRP}}} \hat{c}^{\top}w
\]

where \(S_{\mathrm{CVRP}}\) denotes the set of feasible integer CVRP solutions.

The quality of the resulting decision is evaluated using the true edge costs \(c\), through the decision regret:

\[
R(\hat{c},c)
=
c^\top \hat{w}
-
z^*(c)
\]

where:

\[
z^*(c)=\min_{w\in S_{\mathrm{CVRP}}}c^\top w
\]

The central question is whether minimizing a decision-oriented training objective produces better routing decisions than minimizing prediction error alone.

## 2. Learning Architecture

### Node-Level NLP Encoding

The model uses a shared Transformer encoder to process node-level textual descriptions independently.

- **Text encoder:** DeBERTa-v3-base generates contextualized representations of node-level operational information.
- **Parameter-efficient fine-tuning:** Low-Rank Adaptation (LoRA) is used to adapt the Transformer while reducing the number of trainable parameters.
- **Edge prediction:** A shared multilayer perceptron (MLP) predicts directed edge costs from concatenated source and destination node representations.

For nodes \(i\) and \(j\):

\[
h_i=\operatorname{Encoder}_{\theta}(t_i)
\]

\[
\hat{c}_{ij}
=
\operatorname{MLP}_{\phi}([h_i;h_j])
\]

where \(t_i\) is the textual description associated with node \(i\).

This architecture avoids concatenating all node descriptions into a single Transformer input and allows the same encoder and edge predictor to be applied to graphs with different numbers of nodes. Generalization to larger or previously unseen graphs remains an empirical question.

## 3. Decision-Focused Learning

### MSE Baseline

The standard predict-then-optimize approach trains the model to minimize the Mean Squared Error (MSE) between predicted and true routing costs.

However, minimizing prediction error does not necessarily minimize decision regret: prediction errors can have different consequences depending on their impact on the optimal routing solution.

### SPO+ and Relaxation-Based Training

The project investigates **Smart Predict-and-Optimize (SPO+)**, a convex surrogate loss for decision regret.

For the integer CVRP, the standard SPO+ loss is:

\[
L_{\mathrm{SPO+}}(\hat c,c)
=
-z_I^*(2\hat c-c)
+2\hat c^\top w_I^*(c)
-z_I^*(c)
\]

where \(z_I^*\) and \(w_I^*\) denote optimal objective values and solutions of the integer optimization problem.

Because repeatedly solving a MILP during training can be computationally expensive, the project uses a **hybrid integer–continuous relaxation approach**:

\[
L_{\mathrm{hybrid}}(\hat c,c)
=
-z_L^*(2\hat c-c)
+2\hat c^\top w_I^*(c)
-z_I^*(c)
\]

where:

- \(w_I^*(c)\) and \(z_I^*(c)\) are precomputed using an exact MILP solver.
- \(z_L^*(2\hat c-c)\) is obtained by solving a continuous relaxation during training.
- PyEPO provides the optimization-based subgradients used to train the neural network.

When the continuous feasible region contains the integer feasible set, the hybrid loss remains a convex upper bound on the integer decision regret:

\[
L_{\mathrm{hybrid}}
\geq
L_{\mathrm{SPO+}}
\geq
R(\hat c,c)
\]

This approach reduces the computational cost of optimization calls during training while retaining exact integer solutions as supervision. Its effectiveness depends, among other factors, on the quality of the continuous relaxation.

### Training Strategy

The textual DFL pipeline follows two stages:

1. **MSE warm-up:** Train the NLP model to estimate routing costs using a standard regression objective.
2. **Decision-focused fine-tuning:** Continue training using the hybrid SPO+ loss to account for downstream routing decisions.

Model selection is based on validation decision regret, evaluated using the integer CVRP solver.

## 4. Optimization Models

The optimization component is implemented using **Gurobi** and **PyEPO**.

The current textual DFL pipeline uses:

- **Exact CVRP model:** Mixed-Integer Linear Programming formulation based on Miller–Tucker–Zemlin (MTZ) constraints, used to generate optimal supervision labels and evaluate decision regret.
- **Relaxed CVRP model:** Continuous relaxation of the MTZ formulation, used for optimization-based training.

Additional relaxations, including assignment-based and flow-based formulations, are investigated in the numerical DFL experiments to study the trade-off between computational efficiency and decision quality.

## 5. Experimental Setup and Research Objectives

The project uses synthetic CVRP instances in which edge costs depend on nonlinear transformations of operational conditions.

For the textual experiments, node-level descriptions contain both:

- **Decision-relevant information:** Operational conditions influencing routing costs.
- **Semantic distractors:** Contextual information that does not directly affect the underlying cost function.

The experiments investigate three main questions:

1. **Prediction versus decision quality:** How does decision-focused training compare with MSE in terms of prediction accuracy and exact CVRP regret?
2. **Computational efficiency:** How do different optimization relaxations affect training time and decision quality?
3. **Semantic robustness:** How does the model respond to irrelevant textual information, and can decision-focused training improve robustness to such distractors?

The numerical DFL experiments provide an initial evaluation of the relaxation-based training strategy before its application to Transformer-based textual cost prediction.

## 6. Repository Structure

```text
nlp_spo_plus/
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│       └── textual/
├── models/
│   ├── numerical_dfl/
│   └── textual_dfl/
├── notebooks/
│   ├── numerical_dfl/
│   └── textual_dfl/
├── plots/
│   ├── numerical_dfl/
│   └── textual_dfl/
├── scientific_papers/
├── src/
│   └── nlp_spo_plus/
│       ├── data/
│       ├── models/
│       ├── optimization/
│       └── training/
├── pyproject.toml
├── uv.lock
└── README.md
```

**Main directories:**

- `data/`: Raw, processed, and synthetic CVRP datasets.
- `models/`: Saved model weights and training checkpoints.
- `notebooks/`: Numerical and textual DFL experiments.
- `plots/`: Training metrics and experimental visualizations.
- `scientific_papers/`: Research references.
- `src/nlp_spo_plus/data/`: Dataset generation and preprocessing.
- `src/nlp_spo_plus/models/`: Neural network architectures.
- `src/nlp_spo_plus/optimization/`: Gurobi optimization formulations and PyEPO interfaces.
- `src/nlp_spo_plus/training/`: Training and evaluation pipelines.

## 7. Project Status

This is an ongoing personal research project.

The current work focuses on establishing reliable numerical and textual DFL baselines, evaluating relaxation-based training, and developing reproducible experiments.

Future investigations may include additional optimization formulations, out-of-distribution evaluation, and more extensive analysis of semantic robustness.