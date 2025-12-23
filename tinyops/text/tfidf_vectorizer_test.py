import numpy as np
from tinygrad import Tensor, dtypes
from tinyops.text.tfidf_vectorizer import tfidf_vectorizer
from tinyops._core import assert_close
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest

@pytest.mark.parametrize("corpus", [
    [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
])
def test_tfidf_vectorizer(corpus):
    # sklearn implementation
    vectorizer = TfidfVectorizer()
    expected = vectorizer.fit_transform(corpus).toarray()

    # tinyops implementation
    result = tfidf_vectorizer(corpus)

    # Assert that the result is close to the expected output.
    assert_close(result, Tensor(expected, dtype=dtypes.float32))
