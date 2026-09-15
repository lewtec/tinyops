import re

from tinygrad import Tensor

TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def count_vectorizer(corpus: list[str]) -> Tensor:
    """Convert a collection of text documents to a token count matrix.

    Tokenization uses the pattern that selects tokens of 2+ alphanumeric
    characters after lowercasing.

    Args:
        corpus: List of document strings.

    Returns:
        Document-term matrix (n_documents, n_vocabulary).
    """
    tokens_per_document = [re.findall(TOKEN_PATTERN, document.lower()) for document in corpus]
    vocabulary = sorted(set(token for document_tokens in tokens_per_document for token in document_tokens))
    vocabulary_index = {word: index for index, word in enumerate(vocabulary)}

    counts = [[0] * len(vocabulary) for _ in range(len(corpus))]
    for document_index, document_tokens in enumerate(tokens_per_document):
        for token in document_tokens:
            if token in vocabulary_index:
                counts[document_index][vocabulary_index[token]] += 1

    return Tensor(counts)
