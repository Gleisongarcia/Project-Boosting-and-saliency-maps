import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist():
    """Load MNIST and return the official train/test split."""

    try:
        mnist = fetch_openml(
            name="mnist_784",
            version=1,
            as_frame=False,
            parser="auto"
        )
    except TypeError:
        mnist = fetch_openml(
            name="mnist_784",
            version=1,
            as_frame=False
        )

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(str)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return X_train, X_test, y_train, y_test

def prepare_binary_task(
    X_train, y_train, X_test, y_test, task_name
):
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    if task_name == "3_vs_8":
        train_mask = np.isin(y_train, [3, 8])
        test_mask = np.isin(y_test, [3, 8])

        Xtr = X_train[train_mask]
        Xte = X_test[test_mask]

        ytr = (y_train[train_mask] == 8).astype(int)
        yte = (y_test[test_mask] == 8).astype(int)

        class_names = ["3", "8"]

    elif task_name == "0_4_vs_5_9":
        Xtr, Xte = X_train, X_test

        ytr = (y_train >= 5).astype(int)
        yte = (y_test >= 5).astype(int)

        class_names = ["0–4", "5–9"]

    else:
        raise ValueError("Unknown binary classification task")

    return Xtr, ytr, Xte, yte, class_names
