"""
Training Worker
"""
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def create_random_forest(
    n_estimators: int, criterion: str = "gini", max_depth: Optional[int] = None
) -> RandomForestClassifier:
    """
    Creates an instance of the Random forest classifier.

    :param n_estimators:
    :type n_estimators:
    :param criterion:
    :type criterion:
    :param max_depth:
    :type max_depth:
    :return:
    :rtype:
    """
    if not isinstance(n_estimators, int):
        raise ValueError("Estimators must be integer.")

    if criterion not in ["gini", "entropy", "log_loss"]:
        raise ValueError("Invalid criterion value.")

    if max_depth is not None:
        if not isinstance(max_depth, int):
            raise ValueError("Invalid max depth value.")

    return RandomForestClassifier(
        n_estimators=n_estimators, criterion=criterion, max_depth=max_depth
    )


def create_mlp(
    activation_function: str = "relu",
    solver: str = "adam",
    batch_size: int = 10,
    hidden_layers: Tuple[int, ...] = (4, 4),
) -> MLPClassifier:
    """
    Creates an instance of MLP classifier for training.

    :param activation_function:
    :type activation_function:
    :param solver:
    :type solver:
    :param batch_size:
    :type batch_size:
    :param hidden_layers:
    :type hidden_layers:
    :return:
    :rtype:
    """
    if not isinstance(batch_size, int):
        raise ValueError("Invalid batch size value.")

    if solver not in ["lbfgs", "sgd", "adam"]:
        raise ValueError("Invalid solver value.")

    if activation_function not in ["identity", "logistic", "tanh", "relu"]:
        raise ValueError("Invalid activation function value.")

    if not all(isinstance(x, int) for x in hidden_layers):
        raise ValueError("Invalid hidden layer value.")

    return MLPClassifier(
        activation=activation_function,
        solver=solver,
        batch_size=batch_size,
        hidden_layer_sizes=hidden_layers,
    )


def train(
    x: np.ndarray,
    y: np.ndarray,
    params: dict,
    model: str = "mlp",
) -> str:
    """
    Trains a model and return the location of the model file.

    :param x:
    :type x:
    :param y:
    :type y:
    :param params:
    :type params:
    :param model:
    :type model:
    :return:
    :rtype:
    """

    if model not in ["mlp", "random_forest"]:
        raise ValueError("Invalid model type")

    if model == "mlp":
        create_mlp(
            activation_function=params["activation"],
            solver=params["solver"],
            batch_size=params["batch_size"],
            hidden_layers=params["hidden_layers"],
        )

    else:
        create_random_forest(
            n_estimators=params["n_estimators"],
            criterion=params["criterion"],
            max_depth=params["max_depth"],
        )

    return ""
