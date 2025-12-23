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
