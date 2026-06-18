import math

from tinygrad import Tensor

from tinyops.ops.text.count_vectorizer import count_vectorizer


def tfidf_vectorizer(corpus: list[str]) -> Tensor:
    """Convert documents to a TF-IDF feature matrix.

    Uses smoothed IDF: ``log((N + 1) / (df + 1)) + 1`` and L2
    normalization, matching sklearn defaults.

    Args:
        corpus: List of document strings.

    Returns:
        TF-IDF matrix (n_documents, n_vocabulary).
    """
    counts_tensor = count_vectorizer(corpus)
    counts = counts_tensor.numpy().tolist()
    vocabulary_size = counts_tensor.shape[1]

    document_count = len(corpus)
    document_frequencies = [0] * vocabulary_size
    for term_index in range(vocabulary_size):
        document_frequencies[term_index] = sum(1 for document_counts in counts if document_counts[term_index] > 0)

    inverse_document_frequencies = [
        math.log((document_count + 1) / (document_frequencies[i] + 1)) + 1 for i in range(vocabulary_size)
    ]

    tfidf_rows = []
    for document_counts in counts:
        row = [document_counts[i] * inverse_document_frequencies[i] for i in range(vocabulary_size)]
        row_norm = math.sqrt(sum(value**2 for value in row))
        if row_norm > 0:
            row = [value / row_norm for value in row]
        tfidf_rows.append(row)

    return Tensor(tfidf_rows)
