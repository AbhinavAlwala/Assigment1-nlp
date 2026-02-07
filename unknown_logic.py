def get_word_form(word):
    """
    Returns a pseudo-word class for handling unknowns based on morphology.
    """
    if not word:
        return '<UNK>'

    # Check if number
    if any(c.isdigit() for c in word):
        return '<NUM>'

    # Check suffixes
    if word.endswith('ing'):
        return '<UNK-ING>'
    if word.endswith('ed'):
        return '<UNK-ED>'
    if word.endswith('ly'):
        return '<UNK-LY>'
    if word.endswith('s'):
        return '<UNK-S>'
    if word.endswith('tion'):
        return '<UNK-TION>'
    if word.endswith('er'):
        return '<UNK-ER>'
    if word.endswith('est'):
        return '<UNK-EST>'
    if word.endswith('al'):
        return '<UNK-AL>'
    if word.endswith('ity'):
        return '<UNK-ITY>'
    if word.endswith('y'):
        return '<UNK-Y>'

    # Check capitalization
    if word[0].isupper():
        return '<UNK-CAP>'

    return '<UNK>'


def get_unknown_types():
    """Return the set of unknown token classes used by the model."""
    return {
        '<UNK>', '<NUM>', '<UNK-ING>', '<UNK-ED>', '<UNK-LY>',
        '<UNK-S>', '<UNK-TION>', '<UNK-ER>', '<UNK-EST>',
        '<UNK-AL>', '<UNK-ITY>', '<UNK-Y>', '<UNK-CAP>'
    }

__all__ = ["get_word_form", "get_unknown_types"]
