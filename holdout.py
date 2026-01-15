import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional


def holdout(X,
            Y,
            test_size: float,
            random_state: int
            ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

    if test_size > 0.35:
        warnings.warn(
            "ATTENZIONE: stai utilizzando un training set < 65%. "
            "Questo potrebbe ridurre la capacità di apprendimento del modello.",
            UserWarning
        )

    np.random.seed(random_state)

    n_samples = len(X)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split_point = int(n_samples * (1 - test_size))

    train_idx = indices[:split_point]
    test_idx  = indices[split_point:]

    X_train = X[train_idx]
    X_test  = X[test_idx]
    Y_train = Y[train_idx]
    Y_test  = Y[test_idx]

    return (
        np.array(X_train),
        np.array(X_test),
        np.array(Y_train),
        np.array(Y_test)
    )
