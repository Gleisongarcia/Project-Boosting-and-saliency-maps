from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron


RANDOM_STATE = 42


def make_weak_learners():
    return {
        "Decision stump": DecisionTreeClassifier(
            max_depth=1,
            random_state=RANDOM_STATE
        ),

        "Perceptron": Perceptron(
            max_iter=1000,
            tol=1e-3,
            random_state=RANDOM_STATE
        ),

        "Decision tree depth 3": DecisionTreeClassifier(
            max_depth=3,
            random_state=RANDOM_STATE
        ),

        "Decision tree depth 8": DecisionTreeClassifier(
            max_depth=8,
            random_state=RANDOM_STATE
        )
    }

import inspect
from sklearn.ensemble import AdaBoostClassifier


def build_adaboost(weak_learner, n_estimators):
    parameters = inspect.signature(
        AdaBoostClassifier
    ).parameters

    kwargs = {
        "n_estimators": n_estimators,
        "random_state": RANDOM_STATE
    }

    if "estimator" in parameters:
        return AdaBoostClassifier(
            estimator=weak_learner,
            **kwargs
        )

    return AdaBoostClassifier(
        base_estimator=weak_learner,
        **kwargs
    )
