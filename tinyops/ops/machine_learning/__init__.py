"""Machine learning operations: preprocessing, metrics, classifiers, regressors."""

# Preprocessing / Scaling
from .standard_scaler import standard_scaler
from .min_max_scaler import min_max_scaler
from .max_absolute_scaler import max_absolute_scaler
from .robust_scaler import robust_scaler
from .normalizer import normalizer, NormType
from .binarizer import binarizer
from .one_hot_encoder import one_hot_encoder
from .label_encoder import label_encoder
from .polynomial_features import polynomial_features

# Classification metrics
from .accuracy_score import accuracy_score
from .precision_score import precision_score
from .recall_score import recall_score
from .f1_score import f1_score
from .confusion_matrix import confusion_matrix
from .receiver_operating_characteristic import receiver_operating_characteristic_area

# Regression metrics
from .mean_squared_error import mean_squared_error
from .mean_absolute_error import mean_absolute_error
from .coefficient_of_determination import coefficient_of_determination

# Neighbors
from .nearest_neighbors import nearest_neighbors

# Naive Bayes
from .multinomial_naive_bayes import multinomial_naive_bayes
from .bernoulli_naive_bayes import bernoulli_naive_bayes

# Linear models
from .logistic_regression_step import logistic_regression_step
from .stochastic_gradient_descent_classifier_step import stochastic_gradient_descent_classifier_step
from .stochastic_gradient_descent_regressor_step import stochastic_gradient_descent_regressor_step

# Support vector machines
from .linear_support_vector_classifier import linear_support_vector_classifier
from .linear_support_vector_regressor import linear_support_vector_regressor
from .kernel_support_vector_classifier import kernel_support_vector_classifier, KernelType
from .kernel_support_vector_regressor import kernel_support_vector_regressor

# Trees
from .decision_tree_classifier import decision_tree_classifier
from .decision_tree_regressor import decision_tree_regressor

# Ensembles
from .random_forest_classifier import random_forest_classifier
from .random_forest_regressor import random_forest_regressor
from .adaboost_classifier import adaboost_classifier
from .gradient_boosting_regressor import gradient_boosting_regressor

# Decomposition
from .non_negative_matrix_factorization import non_negative_matrix_factorization
