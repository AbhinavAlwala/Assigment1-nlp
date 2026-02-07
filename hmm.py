import numpy as np
import sys
import os

# --- Helper Functions ---

def parse_conllu(path):
    """
    Parse a .conllu file and return a list of sentences,
    where each sentence is a list of (word, upos) tuples.
    """
    sentences = []
    current = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Sentence boundary
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue

            # Skip comments
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                continue  # malformed line

            token_id = cols[0]
            # Skip multi-word tokens (e.g., "1-2") and empty nodes (e.g., "3.1")
            if "-" in token_id or "." in token_id:
                continue

            form = cols[1]
            upos = cols[3] if len(cols) > 3 else "_"
            current.append((form, upos))

    # Capture last sentence if file doesn't end with a blank line
    if current:
        sentences.append(current)

    return sentences

def tokenize(vocab_list):
    '''
    Create a mapping from item to integer index.
    '''
    # Create dict with indices
    vocab_map = {vocab_list[i]: i for i in range(len(vocab_list))}
    if '<UNK>' not in vocab_map:
        vocab_map['<UNK>'] = len(vocab_map)
    return vocab_map

def get_tokenized_data(sentence_data, tokenized_vocab, tokenized_tags):
    '''
    Converts list of (word, tag) tuples to list of [word_idx, tag_idx] lists.
    '''
    new_data = []
    unk_idx = tokenized_vocab['<UNK>']
    
    for sent in sentence_data:
        new_sent = []
        for word, tag in sent:
            w_idx = tokenized_vocab.get(word, unk_idx)
            t_idx = tokenized_tags.get(tag, -1)
            new_sent.append([w_idx, t_idx])
        new_data.append(new_sent)
    return new_data


# --- Preprocessing & Token Handling ---

def get_word_form(word):
    """
    Returns a pseudo-word class for handling unknowns based on morphology.
    """
    if not word: return '<UNK>'
    
    # Check if number
    if any(c.isdigit() for c in word):
        return '<NUM>'
    
    # Check suffixes
    if word.endswith('ing'): return '<UNK-ING>'
    if word.endswith('ed'): return '<UNK-ED>'
    if word.endswith('ly'): return '<UNK-LY>'
    if word.endswith('s'): return '<UNK-S>'
    if word.endswith('tion'): return '<UNK-TION>'
    if word.endswith('er'): return '<UNK-ER>'
    if word.endswith('est'): return '<UNK-EST>'
    if word.endswith('al'): return '<UNK-AL>'
    if word.endswith('ity'): return '<UNK-ITY>'
    if word.endswith('y'): return '<UNK-Y>'
    
    # Check capitalization
    if word[0].isupper(): return '<UNK-CAP>'
    
    return '<UNK>'

def train_hmm(train_data_path):
    """
    Trains the HMM model from the given conllu file.
    Returns:
        transition_matrix
        emission_matrix
        initial_probs
        tokenized_vocab
        tokenized_tags
    """
    # 1. Load Data
    data = parse_conllu(train_data_path)
    
    # 2. Advanced Vocab Handling
    word_counts = {}
    tags_set = set()
    
    for sent in data:
        for word, tag in sent:
            word_counts[word] = word_counts.get(word, 0) + 1
            tags_set.add(tag)

    # Use all words (min_freq=1) to preserve specific knowledge
    min_freq = 1
    
    vocab_set = set()
    unk_types = set(['<UNK>', '<NUM>', '<UNK-ING>', '<UNK-ED>', '<UNK-LY>', 
                     '<UNK-S>', '<UNK-TION>', '<UNK-ER>', '<UNK-EST>', 
                     '<UNK-AL>', '<UNK-ITY>', '<UNK-Y>', '<UNK-CAP>'])
    
    vocab_set.update(unk_types)
    
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab_set.add(word)
            
    # Listify & Tokenize
    vocab_list = sorted(list(vocab_set))
    tags_list = sorted(list(tags_set))
    
    tokenized_vocab = {v: k for k, v in enumerate(vocab_list)}
    tokenized_tags = {v: k for k, v in enumerate(tags_list)}
    
    vocab_len = len(tokenized_vocab)
    numtags = len(tokenized_tags)
    
    # 3. Convert Data to Indices
    # We keep the logical index conversion simple for transitions
    train_data_indices = []
    
    for sent in data:
        new_sent = []
        for word, tag in sent:
            if word in tokenized_vocab:
                w_idx = tokenized_vocab[word]
            else:
                w_idx = tokenized_vocab['<UNK>'] # Should not happen with min_freq=1
            
            t_idx = tokenized_tags[tag]
            new_sent.append([w_idx, t_idx])
        train_data_indices.append(new_sent)
    
    # 4. Calculate Probabilities
    
    # Smoothing parameters
    alpha = 1e-5 # Smaller alpha
    
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
            # Use threshold 10 to include more data for statistics
            if word_counts[word] <= 10:
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

# --- Viterbi Logic ---

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
        # V[:, 0] = log_pi + log_emission(w0)
        V[:, 0] = log_pi + log_B[w0, :]
        
        # Recursion
        for t in range(1, T):
            wt = sent_indices[t]
            
            # We want max_prev ( V[prev, t-1] + log_A[prev, curr] )
            # V[:, t-1] shape (N,)
            # log_A shape (N, N) where rows=prev, cols=curr
            
            # Broadcast: (N, 1) + (N, N) -> (N, N)
            # Result[i, j] = V[i, t-1] + A[i, j]
            scores = V[:, t-1][:, None] + log_A 
            
            # Max over previous states (axis=0)
            best_prev = np.argmax(scores, axis=0) # shape (N,)
            max_scores = scores[best_prev, np.arange(num_tags)] # shape (N,)
            
            # V[curr, t] = max_prev + log_emission(curr, word)
            V[:, t] = max_scores + log_B[wt, :]
            
            # B[curr, t] = best_prev
            B[:, t] = best_prev
            
        # Termination
        # Select tag with max probability at final timestep
        best_last = np.argmax(V[:, T-1])
        
        # Backtracking
        pred_path = [0] * T
        pred_path[T-1] = best_last
        
        for t in range(T-1, 0, -1):
            # The current tag at t is pred_path[t]
            # Use B[current_tag, t] to find previous tag
            curr_tag_idx = pred_path[t]
            prev_tag_idx = B[curr_tag_idx, t]
            pred_path[t-1] = prev_tag_idx
            
        pred_tags = [idx_to_tag[i] for i in pred_path]
        predicted_sentences.append(pred_tags)
        
    return predicted_sentences, raw_test_data

