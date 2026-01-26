import re

from tinygrad import Tensor


def count_vectorizer(corpus: list[str]) -> Tensor:
    """
    Convert a collection of text documents to a matrix of token counts.

    It produces a sparse representation of the counts using a dense tensor.

    Args:
        corpus: A list of strings (documents).

    Returns:
        A tinygrad Tensor representing the document-term matrix of shape (n_samples, n_features).

    Note:
        The tokenization uses the regex ``r'(?u)\\b\\w\\w+\\b'``. This means it:
        - Converts to lowercase.
        - Selects tokens of 2 or more alphanumeric characters.
        - **Ignores single-character tokens** like "a" and "I".

    Warning:
        This implementation creates a dense matrix (Tensor). For large vocabularies or corpora,
        this can consume significant memory.
    """
    # Tokenize
    tokens = [re.findall(r"(?u)\b\w\w+\b", doc.lower()) for doc in corpus]

    # Build vocabulary
    vocab = sorted(list(set(word for doc_tokens in tokens for word in doc_tokens)))
    vocab_map = {word: i for i, word in enumerate(vocab)}

    # Create count matrix
    counts = [[0] * len(vocab) for _ in range(len(corpus))]
    for i, doc_tokens in enumerate(tokens):
        for token in doc_tokens:
            if token in vocab_map:
                counts[i][vocab_map[token]] += 1

    return Tensor(counts)
