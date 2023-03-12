"""
Training Worker
"""
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


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
    if not batch_size:
        batch_size = "auto"

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
