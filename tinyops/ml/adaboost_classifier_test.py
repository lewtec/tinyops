import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.ml.adaboost_classifier import adaboost_classifier


def test_adaboost_classifier_samme():
    X, y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=0, n_classes=3, random_state=42)

    # Train a scikit-learn AdaBoostClassifier
    base_estimator = DecisionTreeClassifier(max_depth=1)
    n_estimators = 10
    learning_rate = 1.0

    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42,
    )
    model.fit(X, y)

    # Extract parameters from the trained model
    estimators_predictions = np.array([est.predict(X) for est in model.estimators_]).astype(np.float32)
    estimator_weights = model.estimator_weights_.astype(np.float32)
    classes = model.classes_.astype(np.float32)

    # Convert to tinygrad Tensors
    estimators_predictions_tg = Tensor(estimators_predictions)
    estimator_weights_tg = Tensor(estimator_weights)
    classes_tg = Tensor(classes)

    # Get the expected prediction from sklearn
    expected_pred = model.predict(X).astype(np.float32)

    # Get the prediction from the tinyops implementation
    result_tg = adaboost_classifier(
        estimators_predictions_tg,
        estimator_weights_tg,
        classes_tg,
        learning_rate,
    )
    result_tg.realize()

    # Compare the results
    assert_close(result_tg, expected_pred, atol=1e-6, rtol=1e-6)
