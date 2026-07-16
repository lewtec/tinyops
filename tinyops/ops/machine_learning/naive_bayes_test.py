"""Direct tests for naive Bayes shared helpers and classifiers (no reference libs)."""

from tinygrad import Tensor

from tinyops.ops.machine_learning._naive_bayes import (
    BERNOULLI_OUTCOME_COUNT,
    _class_labels_from_posterior_scores,
    _prepare_naive_bayes_training,
)
from tinyops.ops.machine_learning.bernoulli_naive_bayes import bernoulli_naive_bayes
from tinyops.ops.machine_learning.multinomial_naive_bayes import multinomial_naive_bayes


def test_bernoulli_outcome_count_is_binary():
    assert BERNOULLI_OUTCOME_COUNT == 2


def test_prepare_naive_bayes_training_counts():
    features = Tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    labels = Tensor([0, 0, 1, 1])
    classes, class_counts, log_priors, feature_counts = _prepare_naive_bayes_training(features, labels)

    assert classes.tolist() == [0, 1]
    assert class_counts.tolist() == [2.0, 2.0]
    expected_log_prior = float(Tensor(0.5).log().numpy())
    assert abs(float(log_priors[0].numpy()) - expected_log_prior) < 1e-5
    assert abs(float(log_priors[1].numpy()) - expected_log_prior) < 1e-5
    # class 0 saw feature0 twice and feature1 once; class 1 saw feature0 zero times and feature1 once
    assert feature_counts.tolist() == [[2.0, 1.0], [0.0, 1.0]]


def test_class_labels_from_posterior_scores_maps_indices():
    # higher score on class index 1 -> label 7; on index 0 -> label 3
    posteriors = Tensor([[0.1, 0.9], [0.8, 0.2]])
    classes = Tensor([3, 7])
    labels = _class_labels_from_posterior_scores(posteriors, classes)
    assert labels.tolist() == [7, 3]


def test_multinomial_and_bernoulli_agree_on_obvious_cases():
    x_train = Tensor(
        [
            [2.0, 1.0, 0.0, 0.0],
            [3.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
        ]
    )
    y_train = Tensor([0, 0, 1, 1])
    x_test = Tensor([[4.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 4.0]])

    multi = multinomial_naive_bayes(x_train, y_train, x_test)
    assert multi.tolist() == [0, 1]

    x_bin_train = Tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    x_bin_test = Tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    bern = bernoulli_naive_bayes(x_bin_train, y_train, x_bin_test, binarize_threshold=None)
    assert bern.tolist() == [0, 1]


def test_explicit_classes_override_discovery():
    x_train = Tensor([[1.0, 0.0], [0.0, 1.0]])
    y_train = Tensor([0, 1])
    x_test = Tensor([[1.0, 0.0]])
    classes = Tensor([0, 1])
    multi = multinomial_naive_bayes(x_train, y_train, x_test, _classes=classes)
    assert multi.shape == (1,)
