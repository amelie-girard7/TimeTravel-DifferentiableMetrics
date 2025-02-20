# Differentiable Training Objectives (DTO) for Counterfactual Story Rewriting with BART and BARTScore

## 1. Introduction

Counterfactual story rewriting is the task of **modifying an existing story ending** when a new counterfactual event is introduced. The goal is to generate a revised ending that remains both **semantically coherent** and **minimally different** from the original ending while accurately reflecting the new event.

Traditional text generation models operate in a discrete space (using operations like `argmax`), which is **non-differentiable** and makes it difficult to enforce fine-grained semantic constraints. Our approach introduces **Differentiable Training Objectives (DTO)** that allow end-to-end training in the continuous embedding space.

In our method:

- A **trainable BART model** (the rewriting model) generates soft, continuous embeddings.
- A **frozen BART model** (the scorer), downloaded from the [BARTScore repository](https://github.com/neulab/BARTScore), computes a differentiable semantic similarity loss.
- The key idea is to compare the **expected embeddings** (computed by mixing the rewriting model’s probability distribution with the scorer’s input embedding matrix) to the reference (edited) ending embeddings—obtained via a simple lookup in the scorer’s matrix.
- Importantly, the reference sentence is still tokenized and passed as target labels, while the rewriting model outputs are handled in the embedding space.

---
