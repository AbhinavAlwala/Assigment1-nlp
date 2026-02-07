import numpy as np
from tokenization import parse_conllu, build_token_maps, sentences_to_indices
from unknown_logic import get_word_form, get_unknown_types

# --- Helper Functions moved to tokenization.py ---

def train_hmm(train_data_path, min_freq=1, alpha=1e-5, rare_threshold=10):
    """
    Train the HMM model from the given conllu file.

    Parameters:
        train_data_path: Path to training .conllu file
        min_freq: Minimum word frequency to include in vocab (default 1)
        alpha: Additive smoothing constant for initial and transition probabilities (default 1e-5)
        rare_threshold: Frequency threshold for augmenting emissions to unknown classes (default 10)

    Returns:
        transition_matrix, emission_matrix, initial_probs, tokenized_vocab, tokenized_tags
    """
    # 1. Load Data
    data = parse_conllu(train_data_path)
    
    # 2. Build counts
    word_counts = {}
    tags_set = set()
    for sent in data:
        for word, tag in sent:
            word_counts[word] = word_counts.get(word, 0) + 1
            tags_set.add(tag)

    # 3. Tokenization (vocab/tags)
    unk_types = get_unknown_types()
    tokenized_vocab, tokenized_tags = build_token_maps(word_counts, tags_set, min_freq, unk_types)
    vocab_len = len(tokenized_vocab)
    numtags = len(tokenized_tags)

    # 4. Convert Data to Indices
    train_data_indices = sentences_to_indices(data, tokenized_vocab, tokenized_tags)
    
    # 5. Calculate Probabilities
    
    # Smoothing parameters
    # alpha provided via function arg
    
    # Initial Probabilities
    pi = np.zeros(numtags) + alpha 
    total_sentences = 0
    for sentence in train_data_indices:
        if sentence:
            first_tag_idx = sentence[0][1]
            pi[first_tag_idx] += 1
            total_sentences += 1
    pi /= np.sum(pi)

    # Transition Matrix P(tag_j | tag_i)
    trans_mat = np.zeros((numtags, numtags)) + alpha
    
    for i in train_data_indices:
        for j in range(len(i)-1):
            t_curr = i[j][1]
            t_next = i[j+1][1]
            trans_mat[t_curr, t_next] += 1
                
    for i in range(numtags):
        row_sum = sum(trans_mat[i, :])
        trans_mat[i, :] /= row_sum
            
    # Emission Matrix P(word | tag) with Augmented Counting for Rare Words
    # emit_mat[w, t] count(t emits w)
    emit_mat = np.zeros((vocab_len, numtags)) + 1e-5 
    
    idx_unk_types = {k: tokenized_vocab[k] for k in unk_types}
    
    for sent in data:
        for word, tag in sent:
            t_idx = tokenized_tags[tag]
            
            # 1. Standard Count
            if word in tokenized_vocab:
                w_idx = tokenized_vocab[word]
                emit_mat[w_idx, t_idx] += 1
            
            # 2. Augmented Count for Rare Words
            if word_counts[word] <= rare_threshold:
                unk_form = get_word_form(word)
                if unk_form in idx_unk_types:
                    u_idx = idx_unk_types[unk_form]
                    emit_mat[u_idx, t_idx] += 1
                
                # Also add to generic <UNK> to ensuring fallback safety
                k_idx = idx_unk_types['<UNK>']
                emit_mat[k_idx, t_idx] += 1
                    
    # Normalize Emissions
    for j in range(numtags):
        col_sum = sum(emit_mat[:, j])
        emit_mat[:, j] /= col_sum
            
    return trans_mat, emit_mat, pi, tokenized_vocab, tokenized_tags

# Viterbi algorithm removed; now located in viterbi.py

