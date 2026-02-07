import numpy as np
from tokenization import parse_conllu
from unknown_logic import get_word_form


def viterbi_algorithm(transition_matrix, emission_matrix, initial_probs, test_data_path, tokenized_vocab, tokenized_tags):
    """
    Runs Viterbi algorithm on the test data.
    """
    raw_test_data = parse_conllu(test_data_path)

    idx_to_tag = {v: k for k, v in tokenized_tags.items()}

    # Pre-fetch UNK indices
    unk_indices = {}
    for k in tokenized_vocab:
        if k.startswith('<UNK>') or k == '<NUM>':
            unk_indices[k] = tokenized_vocab[k]

    fallback_unk = tokenized_vocab.get('<UNK>', 0)

    num_tags = transition_matrix.shape[0]

    predicted_sentences = []

    epsilon = 1e-12
    log_pi = np.log(initial_probs + epsilon)
    log_A = np.log(transition_matrix + epsilon)
    log_B = np.log(emission_matrix + epsilon)

    for sentence in raw_test_data:
        sent_indices = []
        for form, upos in sentence:
            if form in tokenized_vocab:
                w_idx = tokenized_vocab[form]
            else:
                # Determine which UNK class
                unk_form = get_word_form(form)
                w_idx = unk_indices.get(unk_form, fallback_unk)

            sent_indices.append(w_idx)

        T = len(sent_indices)
        if T == 0:
            predicted_sentences.append([])
            continue

        # Viterbi Matrix V of size (number of tags) x (sentence length)
        # B of same size
        V = np.full((num_tags, T), -np.inf)
        B = np.zeros((num_tags, T), dtype=int)

        # Init
        w0 = sent_indices[0]
        V[:, 0] = log_pi + log_B[w0, :]

        # Recursion
        for t in range(1, T):
            wt = sent_indices[t]
            scores = V[:, t-1][:, None] + log_A
            best_prev = np.argmax(scores, axis=0)
            max_scores = scores[best_prev, np.arange(num_tags)]
            V[:, t] = max_scores + log_B[wt, :]
            B[:, t] = best_prev

        # Termination
        best_last = np.argmax(V[:, T-1])

        # Backtracking
        pred_path = [0] * T
        pred_path[T-1] = best_last
        for t in range(T-1, 0, -1):
            curr_tag_idx = pred_path[t]
            prev_tag_idx = B[curr_tag_idx, t]
            pred_path[t-1] = prev_tag_idx

        pred_tags = [idx_to_tag[i] for i in pred_path]
        predicted_sentences.append(pred_tags)

    return predicted_sentences, raw_test_data

__all__ = ["viterbi_algorithm"]
