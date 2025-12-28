from typing import List
from tinygrad import Tensor
import re

def count_vectorizer(corpus: List[str]) -> Tensor:
    """
    Convert a collection of text documents to a matrix of token counts.

    Args:
        corpus: A list of strings (documents).

    Returns:
        A tinygrad Tensor representing the document-term matrix.
    """
    # Tokenize
    tokens = [re.findall(r'(?u)\b\w\w+\b', doc.lower()) for doc in corpus]

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
