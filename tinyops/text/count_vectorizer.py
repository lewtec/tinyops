from typing import List, Optional
from tinygrad import Tensor
import re
from collections import Counter

def count_vectorizer(corpus: List[str], max_features: Optional[int] = None) -> Tensor:
    """
    Convert a collection of text documents to a matrix of token counts.

    Args:
        corpus: A list of strings (documents).
        max_features: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

    Returns:
        A tinygrad Tensor representing the document-term matrix.
    """
    # Tokenize
    tokens = [re.findall(r'(?u)\b\w\w+\b', doc.lower()) for doc in corpus]

    if max_features is not None:
        # Count term frequencies across the entire corpus
        all_tokens = [word for doc_tokens in tokens for word in doc_tokens]
        counter = Counter(all_tokens)
        # Select top max_features by frequency
        # most_common returns list of (elem, count) sorted by count.
        # Note: ties are broken by insertion order.
        top_words = [word for word, count in counter.most_common(max_features)]
        vocab = sorted(top_words)
    else:
        # Build vocabulary
        vocab = sorted(list(set(word for doc_tokens in tokens for word in doc_tokens)))

    vocab_map = {word: i for i, word in enumerate(vocab)}

    # Create count matrix
    # If vocab is empty (empty corpus), len(vocab) is 0.
    if not vocab:
        return Tensor.zeros(len(corpus), 0)

    counts = [[0] * len(vocab) for _ in range(len(corpus))]
    for i, doc_tokens in enumerate(tokens):
        for token in doc_tokens:
            if token in vocab_map:
                counts[i][vocab_map[token]] += 1

    return Tensor(counts)
