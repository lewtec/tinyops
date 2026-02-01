import math

from tinygrad import Tensor

from tinyops.text.count_vectorizer import count_vectorizer


def tfidf_vectorizer(corpus: list[str]) -> Tensor:
    """
    Convert a collection of text documents to a matrix of TF-IDF features.

    TF-IDF (Term Frequency-Inverse Document Frequency) scales the impact of tokens
    that occur very frequently in a given corpus (like "the", "a") and are therefore
    less informative than features that occur in a small fraction of the training corpus.

    This implementation aligns with scikit-learn's default behavior:
    - Uses ``log((N + 1) / (df + 1)) + 1`` for IDF (smoothing).
    - Applies L2 normalization to the resulting TF-IDF vectors.

    Args:
        corpus: A list of strings (documents).

    Returns:
        A tinygrad Tensor representing the document-term matrix of shape (n_samples, n_features).

    Warning:
        This implementation builds a dense matrix. For large corpora or vocabularies,
        it may consume significant memory.
    """
    counts_tensor = count_vectorizer(corpus)
    counts = counts_tensor.numpy().tolist()
    vocab_len = counts_tensor.shape[1]

    # TF-IDF calculation
    n_docs = len(corpus)
    df = [0] * vocab_len
    for term_idx in range(vocab_len):
        df[term_idx] = sum(1 for doc_counts in counts if doc_counts[term_idx] > 0)

    # Scikit-learn's idf formula (smooth_idf=True): log((n_docs + 1) / (df + 1)) + 1
    idf = [math.log((n_docs + 1) / (df[i] + 1)) + 1 for i in range(vocab_len)]

    # Scikit-learn's tf-idf formula: tf * idf, then L2 normalize
    tfidf = []
    for doc_counts in counts:
        doc_tfidf = [doc_counts[i] * idf[i] for i in range(vocab_len)]
        norm = math.sqrt(sum(x**2 for x in doc_tfidf))
        if norm > 0:
            doc_tfidf = [x / norm for x in doc_tfidf]
        tfidf.append(doc_tfidf)

    return Tensor(tfidf)
