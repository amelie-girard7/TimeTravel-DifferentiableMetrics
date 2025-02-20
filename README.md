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

## 2. Key Steps in the DTO Approach

### Step 1. Two Separate BART Models

- **Rewriting Model (Trainable):**  
  This BART model is fine-tuned for counterfactual rewriting and produces logits. These logits are then converted to probabilities and used to compute expected embeddings.

- **Scorer Model (Frozen):**  
  A separate BART model, downloaded from [BARTScore](https://github.com/neulab/BARTScore), is used solely for computing the loss. Its parameters and input embedding matrix are frozen to ensure stability. Both models share the same vocabulary.

### Step 2. Computing Expected Embeddings

1. **Generate Probability Distributions:**  
   The rewriting model produces logits \(L \in \mathbb{R}^{B \times T \times V}\). Apply softmax:
   \[
   P_{i,j,k} = \frac{\exp(L_{i,j,k})}{\sum_{k'} \exp(L_{i,j,k'})}
   \]

2. **Compute Expected Embeddings:**  
   With the scorer’s input embedding matrix \(E \in \mathbb{R}^{V \times D}\), the expected embedding for token position \(j\) is:
   \[
   \hat{e}_{i,j} = \sum_{k=1}^{V} P_{i,j,k} \cdot E_{k} \quad \Longrightarrow \quad \hat{E} = P \times E
   \]
   **Note:** The scorer’s embedding matrix is detached (frozen) so that gradients do not flow into it.

### Step 3. Obtaining Reference Embeddings

- The reference (edited_ending) is tokenized and its embeddings are obtained via a lookup in the scorer’s input embedding matrix.
- **Important:** The reference remains as token IDs and is passed via the `labels` parameter (as in the original BARTScorer code).

### Step 4. Loss Computation and Backpropagation

- A differentiable loss (e.g., Mean Squared Error) is computed between the expected embeddings \(\hat{E}\) and the reference embeddings:
  \[
  L_{\text{DTO}} = \frac{1}{B \times T \times D} \sum_{i=1}^{B} \sum_{j=1}^{T} \| \hat{e}_{i,j} - e^{\text{ref}}_{i,j} \|^2
  \]
- **Gradient Flow:**  
  - Gradients flow through the rewriting model’s outputs and probabilities.
  - The scorer’s parameters, its embedding matrix, and the reference embeddings are detached so that they remain unchanged.

---

## 3. Implementation Enhancements

### Handling Inputs

- **Reference Sentence:**  
  The reference sentence is still tokenized and passed as labels (using `labels=tgt_tokens`), exactly as in the original implementation (see line 62 of the BARTScorer code).

- **Expected Embeddings as Inputs:**  
  To support passing expected embeddings directly, we introduce a new method `score_embeds` that mimics the existing `score` method but accepts embeddings via the `inputs_embeds` keyword (a native feature of BartModel).

### New `score_embeds` Method

Below is an annotated example of the new method:

```python
def score_embeds(self, inputs_embeds, tgts, batch_size=4):
    """
    Score a batch using direct embeddings as inputs.
    
    Args:
        inputs_embeds (torch.Tensor): Expected embeddings from the rewriting model.
        tgts (List[str]): The reference target sentences (edited_ending), tokenized as usual.
        batch_size (int, optional): Batch size for processing.
    
    Returns:
        List[float]: The computed scores.
    """
    score_list = []
    for i in range(0, len(inputs_embeds), batch_size):
        # Select the current batch of embeddings.
        batch_embeds = inputs_embeds[i: i + batch_size]
        
        # Tokenize the target/reference sentences.
        encoded_tgt = self.tokenizer(
            tgts[i: i + batch_size],
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        tgt_tokens = encoded_tgt['input_ids'].to(self.device)
        
        with torch.no_grad():
            # Feed the precomputed embeddings via inputs_embeds.
            output = self.model(
                inputs_embeds=batch_embeds,
                attention_mask=None,  # Provide or compute a mask if required.
                labels=tgt_tokens     # The target remains token IDs.
            )
            # Compute loss and derive scores (e.g., using negative loss)
            logits = output.logits.view(-1, self.model.config.vocab_size)
            loss = self.loss_fct(logits, tgt_tokens.view(-1))
            score_list += [-x.item() for x in loss]
    
    return score_list
```

### Gradient and Model Freezing

- **Freezing the Scorer:**  
  The scorer model (including its embedding matrix) is frozen so that gradients update only the rewriting model.
- **Detaching References:**  
  The reference embeddings (via tokenization) and the scorer’s input embedding matrix are detached from the computation graph to prevent updates.

---

## 4. Next Steps

- **Integration:**  
  Integrate the new `score_embeds` method into the local version of the BARTScorer package.
  
- **Recompilation:**  
  Recompile the modified package to reflect these changes.
  
- **Validation:**  
  Confirm with Inigo whether the scorer from [BARTScore](https://github.com/neulab/BARTScore) is the only allowed model for scoring.

---

## 5. Recap of Key Equations

1. **Softmax Conversion:**
   \[
   P_{i,j,k} = \frac{\exp(L_{i,j,k})}{\sum_{k'} \exp(L_{i,j,k'})}
   \]
2. **Expected Embeddings:**
   \[
   \hat{e}_{i,j} = \sum_{k=1}^{V} P_{i,j,k} \cdot E_{k} \quad \Rightarrow \quad \hat{E} = P \times E
   \]
3. **DTO Loss (MSE Example):**
   \[
   L_{\text{DTO}} = \frac{1}{B \times T \times D} \sum_{i=1}^{B} \sum_{j=1}^{T} \| \hat{e}_{i,j} - e^{\text{ref}}_{i,j} \|^2
   \]

---
