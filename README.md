# AID-829 Assignment 1: POS Tagging with HMM

This project implements a Part-of-Speech (POS) tagging system using Hidden Markov Models (HMM) and the Viterbi algorithm.

## Structure

```
/AID-829-Assignment1/
│
├── data/
│   ├── en_ewt-ud-train.conllu  <-- Training Dataset
│   └── en_ewt-ud-test.conllu   <-- Test Dataset
│
├── hmm.py           <-- Core logic for Training & Viterbi
├── evaluate.py      <-- Script to run accuracy tests
├── report.pdf       <-- Project Report
└── README.md        <-- This file
```

## How to Run

1.  Ensure you have Python installed.
2.  Install `numpy`:
    ```bash
    pip install numpy
    ```
3.  Run the evaluation script:
    ```bash
    python evaluate.py
    ```

## Logic

-   **Training**: Calculate Transition and Emission probabilities from the training corpus.
-   **Viterbi**: Use the learned probabilities to decode the best tag sequence for the test sentences.
-   **Evaluation**: Compare predicted tags against gold standard tags.
