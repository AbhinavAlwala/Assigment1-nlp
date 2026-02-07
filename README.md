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
├── tokenization.py  <-- Conllu parsing & vocab/tags tokenization helpers
├── unknown_logic.py <-- Unknown token handling (morphology classes)
├── hmm.py           <-- Core logic for Training
├── viterbi.py       <-- Viterbi decoding
├── evaluate.py      <-- Script to run accuracy tests
├── report.pdf       <-- Project Report
└── README.md        <-- This file
```

## How to Run

1.  Ensure you have Python installed.
2.  Install required packages:
    ```bash
    pip install numpy pandas
    ```
3.  Run the evaluation script (defaults use files in `data/`):
    ```bash
    python evaluate.py
    ```
    This script runs with fixed defaults; edit `evaluate.py` if you want to change paths or parameters.

## Logic

-   **Training**: Calculate Transition and Emission probabilities from the training corpus.
-   **Viterbi**: Use the learned probabilities to decode the best tag sequence for the test sentences.
-   **Evaluation**: Compare predicted tags against gold standard tags.

## Notes

-   Parsing (`parse_conllu`) and vocab/tags tokenization helpers are in `tokenization.py`.
-   Unknown token morphology logic (`get_word_form`, `get_unknown_types`) is in `unknown_logic.py`.
-   `hmm.py` trains the model; `viterbi.py` handles decoding.
