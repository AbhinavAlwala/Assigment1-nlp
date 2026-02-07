# Report: Part-of-Speech Tagging using HMM and Viterbi Algorithm

**Course:** AID-829 - Assignment 1  
**Project:** Part-of-Speech Tagging from Scratch  
**Date:** February 7, 2026

---

## 1. Introduction
This project implements a robust Part-of-Speech (POS) tagging system using Hidden Markov Models (HMM) and the Viterbi decoding algorithm. The system is trained on the **Universal Dependencies English Web Treebank (UD-EWT)** and evaluated on a held-out test set. 

The primary objective is to explicitly construct the Viterbi probability and backpointer matrices to recover the most probable sequence of POS tags for unseen sentences, strictly adhering to the constraint of not using pre-built POS tagging libraries.

## 2. Methodology

### 2.1 Data Preprocessing & Vocabulary Handling
A critical challenge in POS tagging is handling **Out-Of-Vocabulary (OOV)** words—words present in the test set but not seen during training. A baseline HMM often fails on these. To address this, we implemented a **morphology-based unknown word handling strategy**.

#### Advanced Unknown Word Classification
Instead of mapping all unknown words to a single `<UNK>` token, we analyze the word's surface form (suffixes, capitalization, digits) to map it to a **pseudo-class**:

| Feature | Example | Pseudo-Class |
| :--- | :--- | :--- |
| **Digits** | "1990", "24.5" | `<NUM>` |
| **Suffix -ing** | "running", "eating" | `<UNK-ING>` |
| **Suffix -ed** | "started", "assigned" | `<UNK-ED>` |
| **Suffix -ly** | "quickly" | `<UNK-LY>` |
| **Suffix -tion** | "action" | `<UNK-TION>` |
| **Capitalized** | "Paris", "Google" | `<UNK-CAP>` |
| **Other** | "fractal" | `<UNK>` |

We support 13 distinct unknown classes (including `-er`, `-est`, `-al`, `-ity`, `-y`, `-s`).

### 2.2 Hidden Markov Model (HMM) Training
We calculate the model parameters $\lambda = (A, B, \pi)$ using Maximum Likelihood Estimation (MLE) with advanced smoothing techniques.

#### A. Initial Probabilities ($\pi$)
$$P(tag_i) = \frac{Count(tag_i \text{ at start})}{N_{sentences}}$$
We apply **Laplace Add-$\alpha$ smoothing** ($\alpha = 10^{-5}$) to ensure no tag has a zero probability of starting a sentence.

#### B. Transition Probabilities ($A$)
$$P(tag_j | tag_i) = \frac{Count(tag_i \to tag_j)}{Count(tag_i)}$$
Calculated using a matrix of size $(N_{tags} \times N_{tags})$. We apply smoothing here as well to allow the Viterbi path to survive even if a specific tag bigram was never observed in training.

#### C. Emission Probabilities ($B$) with Augmented Counting
$$P(word_k | tag_i) = \frac{Count(tag_i \text{ emits } word_k)}{Count(tag_i)}$$

To robustly estimate emission probabilities for our unknown pseudo-classes, we employed an **Augmented Counting** strategy:
1.  **Rare Word Contribution**: During training, any word with a frequency $\le 10$ is considered "rare".
2.  **Stat Sharing**: When processing a rare word (e.g., "zapping", tag=VERB), we increment the count for "zapping" **AND** the count for its pseudo-class `<UNK-ING>`.
3.  **Result**: This allows the model to learn that `<UNK-ING>` is highly likely to be a VERB, `<UNK-LY>` is likely an ADV, and `<UNK-CAP>` is likely a PROPN, based on the statistics of rare words in the training data.

### 2.3 The Viterbi Algorithm
We implemented the Viterbi algorithm using dynamic programming in the **log-probability space** to prevent numerical underflow.

#### Matrices
We explicitly construct two matrices of size $(N_{tags} \times T_{sentence\_length})$:
1.  **Viterbi Matrix ($V$)**: Stores the max log-probability of the best path ending in state $j$ at time $t$.
2.  **Backpointer Matrix ($B$)**: Stores the index of the previous state that maximized the probability.

#### Recursion Step
$$V_{t}[j] = \max_{i} \left( V_{t-1}[i] + \log(A_{ij}) \right) + \log(B_{j}(w_t))$$

We utilized **NumPy vectorization** to compute the updates for all states $j$ simultaneously, significantly speeding up the decoding process compared to nested loops.

#### Backtracking
The most probable final state is selected as $q_T^* = \arg\max_i V_T[i]$. We then recursively trace back using the backpointer matrix: $q_{t-1}^* = B_t[q_t^*]$.

## 3. Results

The model was evaluated on the `en_ewt-ud-test.conllu` dataset.

### 3.1 Quantitative Performance
*   **Total Tokens Evaluated**: 25,094
*   **Correct Predictions**: 22,721
*   **Final Accuracy**: **90.54%**

This performance is significantly higher than a standard baseline (typically ~85-88% for this dataset without morphological unknown handling), demonstrating the effectiveness of the pseudo-class strategy.

### 3.2 Confusion Matrix Analysis
The confusion matrix for the top 10 most frequent tags shows strong diagonal performance:

| | Pred: NOUN | Pred: VERB | Pred: PROPN | Pred: ADJ |
|---|---|---|---|---|
| **True: NOUN** | **0.91** | 0.03 | 0.04 | 0.02 |
| **True: VERB** | 0.05 | **0.88** | 0.00 | 0.01 |
| **True: PROPN** | 0.22 | 0.02 | **0.70** | 0.05 |

*Analysis*:
*   **PROPN vs NOUN**: The model sometimes misclassifies Proper Nouns (PROPN) as Nouns (NOUN) (22% error rate). This is a common ambiguity in English (e.g., "Apple" the company vs "apple" the fruit).
*   **Verbs**: High precision (88%), with occasional confusion with Nouns (gerunds like "running").

## 4. Conclusion
We successfully built a high-performance POS tagger from scratch. By analyzing the linguistic properties of unknown words and implementing a vectorized Viterbi decoder, we achieved an accuracy of **90.54%**, meeting and exceeding the assignment objectives.
