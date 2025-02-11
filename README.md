# Differentiable Training Objectives (DTO) for Counterfactual Story Rewriting with BART and BARTScore

---

## 1. Introduction

Counterfactual story rewriting is the task of **modifying an existing story ending** based on a newly introduced **counterfactual event** while ensuring that the revised ending maintains **semantic coherence** and **minimal necessary edits**.

Traditional text generation models optimize for **fluency**, but they often lack explicit constraints that ensure the new ending remains **minimally different** from the original while accurately reflecting the counterfactual event. To solve this, our approach introduces **Differentiable Training Objectives (DTO)**. In our method:

- **BART** is used as the generator but produces **continuous (soft) embeddings** rather than hard token selections.
- **BARTScore** is employed as a differentiable loss function that measures the semantic quality of the rewritten counterfactual endings.
- The model retains **end-to-end differentiability** by passing soft probability distributions through BART’s embedding space—enabling gradient-based optimization.

---

## 2. Why Differentiable Training with BARTScore?

### Limitations of Discrete Token-Based Models
Traditional sequence-to-sequence models generate text by selecting tokens via the `argmax` operation. This process is inherently **non-differentiable** because only a single token is chosen (hard decision) at every time step. As a consequence, gradient-based training methods cannot propagate errors through these discrete choices.

### Advantages of the DTO Approach

1. **Soft Token Representations**  
   - Instead of picking one token, the model maintains a **soft probability distribution** over the entire vocabulary.
   - These probabilities are used to compute a **weighted sum** over BART’s embedding space. This “expected embedding” represents the token in a continuous form and allows gradients to flow seamlessly through the network.

2. **Using BARTScore as a Loss Function**  
   - BARTScore measures the **semantic similarity** between the generated (edited) ending and the reference.
   - By using BARTScore as a differentiable loss function, the model directly optimizes for outputs that are contextually and semantically aligned with the ground truth.

3. **Freezing the BARTScore Model**  
   - The BARTScore model is kept **frozen** during training to ensure that it acts as a stable evaluator and does not adjust during the optimization of the generator.
   - This separation guarantees that BARTScore remains a pure metric for semantic similarity rather than influencing the generation process.

---

## 3. How BART Embeddings Work

BART embeddings are central to achieving smooth gradient propagation. Here’s an in-depth look at the two approaches:

### A. Traditional Token-Based Approaches

In standard sequence-to-sequence models:
1. The model predicts a probability distribution over the vocabulary at each timestep.
2. The token with the highest probability is selected using `argmax`.
3. This discrete token is then used as the input for the next time step.

> **Issue:** The `argmax` operation introduces a non-differentiable step that prevents gradients from flowing back through the entire sequence.

### B. Soft Token Representation via BART Embeddings

In our DTO approach, instead of selecting a single token, we compute a **soft representation**:
1. The model computes a probability distribution over all possible next tokens.
2. These probabilities are used to compute a **weighted sum** over the token embeddings.  
   For example, suppose the model predicts for a token position:
   ```math
   P(run) = 0.5,\ P(walk) = 0.3,\ P(jump) = 0.2
   ```
   And the token embeddings are:
   - **Emb(run)** = [1, 0, 0]  
   - **Emb(walk)** = [0, 1, 0]  
   - **Emb(jump)** = [0, 0, 1]  

   The expected embedding is then:
   ```math
   E = (0.5 × [1, 0, 0]) + (0.3 × [0, 1, 0]) + (0.2 × [0, 0, 1]) = [0.5, 0.3, 0.2]
   ```
3. This **expected embedding** is then fed into subsequent layers.  
   **Advantage:** The use of soft embeddings preserves uncertainty and—critically—enables smooth gradient propagation throughout the model.

### C. BART Embeddings in Counterfactual Story Rewriting

For counterfactual story rewriting:
1. The BART model generates a probability distribution over all possible rewritten endings.
2. **Soft embeddings** are computed using the weighted sum method rather than selecting hard tokens.
3. These continuous representations are compared to the ground-truth edited endings using **BARTScore**.
4. **Gradients flow through the entire process**, enabling direct optimization of text similarity.

> **Benefit:** This results in **end-to-end differentiability**, which leads to more stable and efficient training.

---

## 4. Task Definition

### Counterfactual Story Rewriting

Given:
- **Premise (`X_P`)**: The background or context of the story.
- **Initial Event (`X_{IE}`)**: The original event in the story.
- **Counterfactual Event (`X_{CE}`)**: The event that replaces the initial event.
- **Original Ending (`X_{OE}`)**: The original conclusion of the story.

The goal is to generate a revised ending (`ŷ_{EE}`) that:
1. **Reflects the Counterfactual Event (`X_{CE}`)**
2. **Maintains Coherence with the Premise (`X_P`)**
3. **Minimally Deviates from the Original Ending (`X_{OE}`)**

#### Input-Output Structure

Each training instance includes:
- **Premise (`X_P`)**
- **Initial Event (`X_{IE}`)**
- **Counterfactual Event (`X_{CE}`)**
- **Original Ending (`X_{OE}`)**
- **Edited Ending (`Y_{EE}`)**: The rewritten ending reflecting `X_{CE}`.

**Example:**

| **Initial Scenario**                                                     | **Counterfactual Scenario**                                             |
|--------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Premise:** John has a severe headache.                                 | **Premise:** John has a severe headache.                                |
| **Initial Event:** He takes *two aspirin pills*.                         | **Counterfactual Event:** He takes a *new experimental pill*.           |
| **Original Ending:** The aspirin takes *a few hours* to work.            | **Edited Ending:** The experimental pill works *within minutes*.        |

---

## 5. Mathematical Formulation

DTO combines **label-smoothed cross-entropy** with a **BARTScore-based differentiable loss**.

### 1. Label-Smoothed Cross-Entropy

The standard cross-entropy loss for Maximum Likelihood Estimation (MLE) is defined as:

```math
L_{MLE} = -\sum_{t=1}^{T} \log P_θ(y_t | y_{<t}, X_P, X_{IE}, X_{CE}, X_{OE})
```

Label smoothing is then applied:
```math
L_{Smooth} = (1-ε) L_{MLE} + ε L_{Aux}
```
where `L_{Aux}` is a uniform penalty term and `ε` is the smoothing hyperparameter.

### 2. Differentiable BARTScore Loss

The overall DTO loss is defined as:
```math
L_{DTO} = L_{Smooth} + α ⋅ L_{BART}
```
where:
```math
L_{BART} = - BARTScore(ŷ_{EE}, Y_{EE})
```
- `ŷ_{EE}` is the generated (edited) ending.
- `Y_{EE}` is the reference edited ending.

Minimizing this loss encourages the model to generate outputs that are semantically similar to the ground truth.

---

## 6. Implementation in Code

The DTO approach is implemented across several modules:

- **`src/main.py`**:  
  - Sets up training (both MLE and DTO modes) using PyTorch Lightning.
  - Configures logging (e.g., with WandB), checkpointing, and evaluation.
  - Handles data loading and model instantiation.
  
- **`src/models/model.py`**:  
  - Defines the `FlanT5FineTuner` LightningModule.
  - In **MLE mode**, it computes the standard cross-entropy loss.
  - In **DTO mode**, it generates soft embeddings (using a softmax-weighted sum over the embedding matrix) and computes a differentiable DTO loss using BARTScore.
  
- **`src/utils/metrics.py`**:  
  - Implements the `MetricsEvaluator` class, which wraps BARTScore.
  - Provides functions to compute semantic similarity scores between generated texts and reference texts.

### Interactive Example: Soft Embedding Computation

Suppose the model outputs a probability distribution for a given token:
```math
P(run) = 0.5,\ P(walk) = 0.3,\ P(jump) = 0.2
```
and the corresponding embeddings are:
- **Emb(run)** = [1, 0, 0]
- **Emb(walk)** = [0, 1, 0]
- **Emb(jump)** = [0, 0, 1]

The expected (soft) embedding is calculated as:
```math
E = 0.5 × [1, 0, 0] + 0.3 × [0, 1, 0] + 0.2 × [0, 0, 1] = [0.5, 0.3, 0.2]
```
This expected embedding is then passed through the subsequent layers, allowing gradients to propagate through the probabilities.

### Running the Code

1. **Install Dependencies:**  
   Ensure you have installed the required packages such as PyTorch, Hugging Face Transformers, PyTorch Lightning, and WandB. See the [requirements.txt](#) file for details.

2. **Data Preparation:**  
   Prepare your dataset with the required fields (Premise, Initial Event, Counterfactual Event, Original Ending, Edited Ending).

3. **Training:**  
   - For **MLE training**, enable `mle_enabled` in the configuration.
   - For **DTO training**, enable `dto_enabled` (or `use_differentiable_metrics`) in the configuration.
   - Run the training script:
     ```bash
     python src/main.py
     ```

4. **Evaluation:**  
   The script automatically performs evaluation on both validation and test sets, computes BARTScore metrics, and saves detailed CSV files with evaluation results.

---

## 7. File Structure

```
TimeTravel-DifferentiableMetrics/
├── src/
│   ├── main.py                 # Training and evaluation setup
│   ├── models/
│   │   └── model.py            # FlanT5FineTuner definition (supports both DTO and MLE)
│   ├── data_loader.py          # Data loading utilities
│   ├── utils/
│   │   ├── config.py           # Configuration parameters
│   │   └── metrics.py          # MetricsEvaluator (BARTScore wrapper)
│   └── BARTScore_metric/
│       └── bart_score.py       # BARTScore implementation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 8. Conclusion

The Differentiable Training Objectives (DTO) approach allows counterfactual story rewriting models to be trained end-to-end by leveraging soft token representations and a differentiable BARTScore loss. This results in models that are both semantically coherent and efficient to train using gradient-based optimization.

For further details, please refer to the inline comments in the code and the interactive examples provided above.

