# NLP-Driven Decision-Focused Learning for CVRP

This repository contains the implementation of an end-to-end **Predict-and-Optimize (ML4CO)** pipeline that bridges Natural Language Processing (NLP) and Operations Research (OR). 

The goal of this project is to infer edge routing costs for a **Capacitated Vehicle Routing Problem (CVRP)** directly from unstructured, noisy textual data (e.g., local weather, traffic updates, events), and to optimize the vehicle routes by backpropagating combinatorial decision regret directly into a Transformer encoder.

##  Scientific Methodology

Traditional machine learning models predict objective coefficients (costs) by minimizing standard regression metrics like Mean Squared Error (MSE). However, MSE treats all prediction errors equally, ignoring the downstream optimization task. A small MSE error on a critical edge can lead to drastically suboptimal routing decisions.

This project implements **Decision-Focused Learning (DFL)** using the **SPO+ (Smart Predict and Optimize)** surrogate loss function, aligning the NLP gradients directly with the combinatorial routing objective.

### 1. Siamese Node-Level NLP Architecture
To ensure scalability and avoid context-length bottlenecks (Transformer $O(L^2)$ complexity), we avoid concatenating the entire graph's text. Instead, we utilize a scalable Siamese architecture:
* **Node Encoder:** A `DeBERTa-v3` base model processes each node's local text independently, extracting a rich latent semantic vector $h_i \in \mathbb{R}^{768}$.
* **Parameter-Efficient Fine-Tuning:** The encoder is fine-tuned using **LoRA** (Low-Rank Adaptation), targeting the disentangled attention weights (`query_proj`, `value_proj`) to adapt to the routing task while keeping the VRAM footprint minimal.
* **Edge Prediction Head:** Directed edge costs $c_{ij}$ are predicted by concatenating source and target embeddings $[h_i \oplus h_j]$ and passing them through an MLP, granting zero-shot size transferability to larger $N$-node graphs.

### 2. Exact Combinatorial Optimization (PyEPO & Gurobi)
* The CVRP is modeled using the **Miller-Tucker-Zemlin (MTZ)** subtour elimination formulation.
* The exact MILP solver (**Gurobi**) is integrated directly within the PyTorch training loop via **PyEPO**. 
* During the backward pass, PyEPO computes the optimal decisions based on perturbed costs, enabling the computation of the SPO+ subgradients.

##  Repository Structure

```text
nlp_spo_plus/
├── data/                  # Git-ignored: Contains local LLM (Qwen) generated texts and Gurobi w*(c) solutions
├── models/                # Git-ignored: Saved LoRA weights and PyTorch checkpoints
├── notebooks/
│   ├── numerical_dfl/     # Baselines using standard numerical latent features
│   └── textual_dfl/       # End-to-end DeBERTa + PyEPO training workflows
├── src/nlp_spo_plus/      # Core Python package
│   ├── data/              # CVRP data generation, LLM prompting, and PyTorch Datasets
│   ├── models/            # DeBERTa-v3 + LoRA and Edge-Head network definitions
│   └── optimization/      # PyEPO-compliant Gurobi MTZ formulations for CVRP
└── README.md
