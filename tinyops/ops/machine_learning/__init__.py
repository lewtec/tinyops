"""Machine learning operations: preprocessing, metrics, classifiers, regressors."""

# Preprocessing / Scaling
# Classification metrics
from .accuracy_score import accuracy_score
from .adaboost_classifier import adaboost_classifier
from .bernoulli_naive_bayes import bernoulli_naive_bayes
from .binarizer import binarizer
from .coefficient_of_determination import coefficient_of_determination
from .confusion_matrix import confusion_matrix

# Trees
from .decision_tree_classifier import decision_tree_classifier
from .decision_tree_regressor import decision_tree_regressor
from .f1_score import f1_score
from .gradient_boosting_regressor import gradient_boosting_regressor
from .kernel_support_vector_classifier import KernelType, kernel_support_vector_classifier
from .kernel_support_vector_regressor import kernel_support_vector_regressor
from .label_encoder import label_encoder

# Support vector machines
from .linear_support_vector_classifier import linear_support_vector_classifier
from .linear_support_vector_regressor import linear_support_vector_regressor

# Linear models
from .logistic_regression_step import logistic_regression_step
from .max_absolute_scaler import max_absolute_scaler
from .mean_absolute_error import mean_absolute_error

# Regression metrics
from .mean_squared_error import mean_squared_error
from .min_max_scaler import min_max_scaler

# Naive Bayes
from .multinomial_naive_bayes import multinomial_naive_bayes

# Neighbors
from .nearest_neighbors import nearest_neighbors

# Decomposition
from .non_negative_matrix_factorization import non_negative_matrix_factorization
from .normalizer import NormType, normalizer
from .one_hot_encoder import one_hot_encoder
from .polynomial_features import polynomial_features
from .precision_score import precision_score

# Ensembles
from .random_forest_classifier import random_forest_classifier
from .random_forest_regressor import random_forest_regressor
from .recall_score import recall_score
from .receiver_operating_characteristic import receiver_operating_characteristic_area
from .robust_scaler import robust_scaler
from .standard_scaler import standard_scaler
from .stochastic_gradient_descent_classifier_step import stochastic_gradient_descent_classifier_step
from .stochastic_gradient_descent_regressor_step import stochastic_gradient_descent_regressor_step
