import numpy as np
from tinygrad import Tensor, dtypes
from tinyops.text.count_vectorizer import count_vectorizer
from tinyops._core import assert_close
from sklearn.feature_extraction.text import CountVectorizer
import pytest

@pytest.mark.parametrize("corpus", [
    [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
])
def test_count_vectorizer(corpus):
    # sklearn implementation
    vectorizer = CountVectorizer()
    expected = vectorizer.fit_transform(corpus).toarray()

    # tinyops implementation
    result = count_vectorizer(corpus)

    # Assert that the result is close to the expected output.
    # Sklearn returns int64, so we cast our result to the same for comparison.
    assert_close(result.cast(dtypes.int64), Tensor(expected))

def test_count_vectorizer_max_features_distinct():
    # Corpus with distinct term frequencies to avoid tie-breaking issues
    # Words must be >= 2 characters to be picked up by default token pattern
    corpus = [
        "aa aa aa aa", # aa: 4
        "bb bb bb",    # bb: 3
        "cc cc",       # cc: 2
        "dd"           # dd: 1
    ]

    # Test max_features=2 -> should keep 'aa' and 'bb'
    vectorizer = CountVectorizer(max_features=2)
    expected = vectorizer.fit_transform(corpus).toarray()

    result = count_vectorizer(corpus, max_features=2)

    assert_close(result.cast(dtypes.int64), Tensor(expected))

    # Verify vocab size
    assert result.shape[1] == 2
    # Verify we got 'aa' and 'bb'. 'aa' should be first col (sorted), 'bb' second.
    # aa counts: 4, 0, 0, 0
    # bb counts: 0, 3, 0, 0

    expected_tensor = Tensor([
        [4, 0],
        [0, 3],
        [0, 0],
        [0, 0]
    ])
    assert_close(result.cast(dtypes.int64), expected_tensor)

def test_count_vectorizer_max_features_limit():
    # Test that max_features limits the vocabulary size even if it's smaller than total unique words
    corpus = [
        "word1 word2 word3 word4 word5"
    ]
    # 5 unique words. max_features=3.
    result = count_vectorizer(corpus, max_features=3)
    assert result.shape[1] == 3

    # Test max_features > total words
    result_large = count_vectorizer(corpus, max_features=10)
    assert result_large.shape[1] == 5
