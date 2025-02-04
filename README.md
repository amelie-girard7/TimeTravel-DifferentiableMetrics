### Requirement Articulation for Enhancing Counterfactual Story Rewriting via Policy Gradient Optimization

#### **Objective**
To enhance the performance of counterfactual story rewriting by integrating Reinforcement Learning (RL) using Policy Gradient (PG) optimization alongside conventional Maximum Likelihood Estimation (MLE) training. The goal is to improve story coherence, contextual relevance, and alignment with specified counterfactual events in the rewritten story endings.

---

#### **Key Goals**
1. **Baseline Performance Evaluation**: Train and evaluate the model using only MLE to establish baseline performance.
2. **Fine-Tuning with Policy Gradient**: Integrate PG training to optimize the model for coherence, fluency, and adherence to counterfactual constraints.
3. **Metric-Driven Rewards**: Design and implement a robust reward system using multiple evaluation metrics to guide PG optimization.
4. **Comparison of Training Paradigms**: Analyze the differences in performance between pure MLE training and a combined MLE + PG approach.

---

#### **Scope of Experiments**

1. **Experiment 1: Pure MLE Training**
   - Train the model for 6 epochs using cross-entropy loss.
   - Select the best-performing model based on validation loss.

2. **Experiment 2: Combined MLE + PG Training**
   - **Phase 1**: Perform MLE training for 3 epochs and save intermediate checkpoints.
   - **Phase 2**: Fine-tune the MLE-trained model using PG optimization for an additional 3 epochs, with a focus on reward maximization.

---

#### **Task Definition**

The **Counterfactual Story Rewriting Task** involves creating alternative story endings based on a hypothetical counterfactual event. The rewritten ending should:
1. **Incorporate the counterfactual event** seamlessly into the story.
2. **Maintain coherence** with the original premise and initial context.
3. **Minimize unnecessary changes** to the original ending while adapting to the counterfactual event.

---

#### **Dataset and Annotation**

- **Input Format**:
  - A concatenated sequence consisting of:
    1. **Premise**: Foundation of the story context.
    2. **Initial Event**: The event leading to the original story ending.
    3. **Original Ending**: The story's unaltered conclusion.
    4. **Counterfactual Event**: The hypothetical scenario contradicting the initial event.
    
- **Output Format**:
  - The **Edited Ending**, which integrates the counterfactual event into the story.

- **Usage**:
  - MLE training uses these pairs for supervised learning.
  - PG training uses the reference endings to calculate rewards, encouraging the model to align generated text with the annotated targets.

---

#### **Model and Training Design**

1. **Model Architecture**:
   - The T5 transformer model is used for text generation.
   
2. **Training Phases**:
   - **MLE Training**: Supervised learning with cross-entropy loss.
   - **PG Fine-Tuning**: Reward-based optimization using RL to refine the outputs.

---

#### **Reward Metrics for PG Optimization**

1. **ROUGE-L**:
   - Measures structural similarity between generated and reference texts.
2. **BERTScore**:
   - Captures semantic similarity through token embeddings.
3. **BARTScore**:
   - Evaluates fluency and logical sequence alignment.
4. **SacreBLEU**:
   - Measures precision in capturing key phrases and n-grams.
5. **Counterfactual Reward Metrics (CRMs)**:
   - Task-specific metrics to measure counterfactual alignment and minimal deviation from the original ending.

---

#### **Training Workflow for PG Optimization**

1. **Input Preparation**:
   - Tokenize and prepare the concatenated input sequence for model processing.
   
2. **Forward Pass**:
   - Generate logits for token probabilities and compute log probabilities for sequence generation.

3. **Reward Calculation**:
   - Use the defined metrics to assign reward scores to generated outputs.
   
4. **Loss Computation**:
   - Compute policy gradient loss by combining log probabilities and rewards.

---

#### **Deliverables**

1. A detailed comparative analysis of:
   - Performance of MLE-only training.
   - Improvements achieved with combined MLE + PG training.
   
2. Visualization of reward optimization trends during PG training.

3. Insights into the impact of each metric on counterfactual alignment, coherence, and fluency.

---

#### **Success Criteria**

- Quantifiable improvement in metrics like ROUGE-L, BERTScore, and Counterfactual Reward Metrics for edited story endings in PG fine-tuning.
- Human evaluation showing enhanced coherence and adherence to counterfactual scenarios in generated outputs.
---

#### **Todo**

1. The ablated TimeTravel experiment:
Input: premise, initial, counterfactual; Output: edited ending

2. The ART experiment:
First, you have to prepare the dataset by browsing the label in the ART dataset to see which hypothesis is correct and which is incorrect for each sample.
Then:
Input: obs1, incorrect_hypothesis, correct_hypothesis; Output: obs2.
Input: premise, incorrect_hypothesis, correct_hypothesis; Output: obs2.

As models, only one model each for MLE3, MLE6, and MLE3+PG3; the one with the best configuration from the main experiment.