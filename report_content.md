# Report: Part-of-Speech Tagging using HMM and Viterbi Algorithm

**Course:** AID-829 - Assignment 1  

## 1. Introduction
This project implements a Part-of-Speech (POS) tagging system from scratch using Hidden Markov Models (HMM) and the Viterbi decoding algorithm. The system is trained on the Universal Dependencies English Web Treebank (UD-EWT) and evaluated on a held-out test set. The objective is to assign the most probable sequence of POS tags to a given sequence of words.

## 2. Methodology

### 2.1 Hidden Markov Model (HMM)
The problem is modeled as an HMM where:
-   **Hidden States**: Universal POS tags (e.g., NOUN, VERB, ADJ).
-   **Observations**: Words in the sentence.

We estimate the model parameters from the training corpus using Maximum Likelihood Estimation (MLE) with smoothing.

#### Parameter Estimation:
1.  **Initial Probabilities ($\pi$)**:
    $$P(tag) = \frac{Count(tag \text{ at start of sentence})}{Total \text{ Sentences}}$$
    We apply Laplace (Add-$\alpha$) smoothing with $\alpha = 10^{-5}$ to avoid zero probabilities for unseen start tags.

2.  **Transition Probabilities ($A$)**:
    $$P(tag_j | tag_i) = \frac{Count(tag_i \to tag_j)}{Count(tag_i)}$$
    Smoothing is applied to ensure that no transition has zero probability, allowing the Viterbi path to continue even if a specific tag sequence was not observed in training.

3.  **Emission Probabilities ($B$)**:
    $$P(word | tag) = \frac{Count(tag \text{ emits } word)}{Count(tag)}$$
    
    *Handling Unknown Words:*
    One of the key challenges in POS tagging is handling Out-Of-Vocabulary (OOV) words. We implemented a robust strategy involving morphological feature extraction:
    -   We categorize unknown words into pseudo-classes based on suffixes (e.g., *ing*, *ed*, *ly*, *tion*) and capitalization.
    -   Pseudo-classes include: `<UNK-ING>`, `<UNK-ED>`, `<UNK-CAP>`, `<NUM>`, etc.
    -   During training, we augment the counts of these pseudo-classes by "borrowing" statistical strength from low-frequency words (frequency $\le 10$) in the training set. If a training word is rare, we count it towards its specific pseudo-class emission probability.
    -   During testing, any unknown word is mapped to its corresponding pseudo-class, allowing reasonable tag prediction based on morphology.

### 2.2 Viterbi Algorithm
We explicitly constructed the Viterbi matrix $V$ and Backpointer matrix $B$ to decode the best sequence.

-   **Matrix Dimensions**: $(N_{tags} \times T_{length})$.
-   **Log-Probabilities**: To prevent numerical underflow (since probabilities get extremely small when multiplied over long sentences), we performed calculations in the log-domain:
    $$V_t[j] = \max_i (V_{t-1}[i] + \log A_{ij}) + \log B_j(w_t)$$
-   **Backtracking**: We trace back from the best final state using the Backpointer matrix $B$ to recover the optimal tag sequence.

## 3. Results

The model was evaluated on the provided `en_ewt-ud-test.conllu` dataset.

*   **Total Tokens**: 25,094
*   **Correctly Predicted**: 22,721
*   **Accuracy**: **90.54%**

## 4. Discussion
The baseline HMM without sophisticated unknown word handling typically achieves lower accuracy. By implementing morphological classes for unknown words and using all training data (including hapax legomena) to estimate parameters, we significantly improved the model's robustness. The Viterbi algorithm efficiently found the globally optimal tag sequence for each sentence given the learned parameters.

## 5. Conclusion
We successfully implemented a POS tagger from scratch achieving >90% accuracy. The implementation strictly adheres to the constraints (no pre-built taggers) and explicitly constructs the required Viterbi matrices.
