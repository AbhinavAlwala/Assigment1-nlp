import os


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


def build_token_maps(word_counts, tags_set, min_freq, unk_types):
    """Create tokenized vocab and tags dictionaries with unknown types included."""
    vocab_set = set()
    vocab_set.update(unk_types)
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab_set.add(word)

    vocab_list = sorted(list(vocab_set))
    tags_list = sorted(list(tags_set))

    tokenized_vocab = {v: k for k, v in enumerate(vocab_list)}
    tokenized_tags = {v: k for k, v in enumerate(tags_list)}

    return tokenized_vocab, tokenized_tags


def sentences_to_indices(data, tokenized_vocab, tokenized_tags):
    """Convert sentences of (word, tag) to index pairs using provided maps."""
    train_data_indices = []
    for sent in data:
        new_sent = []
        for word, tag in sent:
            w_idx = tokenized_vocab.get(word, tokenized_vocab.get('<UNK>', 0))
            t_idx = tokenized_tags[tag]
            new_sent.append([w_idx, t_idx])
        train_data_indices.append(new_sent)
    return train_data_indices

__all__ = ["parse_conllu", "build_token_maps", "sentences_to_indices"]
