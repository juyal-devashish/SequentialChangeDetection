# data_utils.py

import numpy as np
from typing import Any, Callable, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



class HistoricalDataset:
    """
    Wraps a raw dataset (X, y) and remaps labels so that
    `pre_change_label` → 0 and all others become 1…L.

    Attributes
    ----------
    X : np.ndarray
        Feature matrix or array of inputs.
    y : np.ndarray
        Remapped labels in {0,…,L}.
    num_classes : int
        Number of distinct labels L+1.
    label_map : dict
        Mapping from original labels to remapped ones.
    """
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 pre_change_label: Any,
                 transforms: Optional[Callable[[Any], Any]] = None):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, ...)
            Inputs.
        y : array-like, shape (n_samples,)
            Original labels.
        pre_change_label : scalar
            The label in `y` that should be treated as class 0.
        transforms : callable, optional
            A function `x→x'` to apply to each input on the fly.
        """
        self.X_raw = np.asarray(X)
        self.y_raw = np.asarray(y)
        self.transforms = transforms

        # Build remapping so that pre_change_label→0, others→1…L
        unique_labels = sorted(set(self.y_raw))
        if pre_change_label not in unique_labels:
            raise ValueError(f"{pre_change_label} not in provided labels")
        unique_labels.remove(pre_change_label)
        self.label_map = {pre_change_label: 0,
                          **{lab: i+1 for i, lab in enumerate(unique_labels)}}

        # Apply remapping
        self.y = np.array([self.label_map[lab] for lab in self.y_raw], dtype=int)
        self.num_classes = len(self.label_map)

        # Optionally apply transforms now or later
        if self.transforms is not None:
            self.X = np.array([self.transforms(x) for x in self.X_raw])
        else:
            self.X = self.X_raw

    def get_splits(self,
                   train_frac: float = 0.8,
                   random_state: Optional[int] = None
                  ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                             Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a train/validation split of (X, y).

        Parameters
        ----------
        train_frac : float
            Fraction of data to use for training.
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        (X_train, y_train), (X_val, y_val)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y,
            train_size=train_frac,
            stratify=self.y,
            random_state=random_state
        )
        return (X_train, y_train), (X_val, y_val)


def train_sklearn_classifier(dataset: HistoricalDataset,
                             model: Any
                            ) -> Any:
    """
    Fits a scikit-learn–style model on the training portion
    of `dataset`, then returns the fitted model.

    Parameters
    ----------
    dataset : HistoricalDataset
        Must have .get_splits() returning ((X_train, y_train), ...)
    model : object
        Any estimator implementing .fit(X, y).

    Returns
    -------
    model : object
        The fitted estimator (model.fit() was called in place).
    """
    (X_train, y_train), (X_val, y_val) = dataset.get_splits()
    model.fit(X_train, y_train)
    return model


def generate_synthetic_data(n_samples: int = 1000,
                            n_features: int = 2,
                            n_classes: int = 3,
                            cluster_std: float = 1.0,
                            center_box: float = 10.0,
                            class_sep: float = 1.0,
                            weights: Optional[np.ndarray] = None,
                            random_state: int = None):
    """
    Creates X ∈ R^{n_samples×n_features}, y ∈ {0…n_classes-1} by sampling
    from Gaussian clusters with controllable separation, variance, and class priors.

    Parameters
    ----------
    n_samples   : total number of points
    n_features  : dimensionality of each point
    n_classes   : number of clusters / classes
    cluster_std : within-cluster standard deviation
    center_box  : scale to sample each cluster center uniformly from
                  [-center_box, +center_box]^d
    class_sep   : multiplier on the distance between cluster centers
                  (larger = easier problem)
    weights     : array of length n_classes summing to 1, giving class priors;
                  if None, defaults to uniform
    random_state: seed for reproducibility

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), dtype=int
    """
    rng = np.random.RandomState(random_state)

    # 1) Sample cluster centers in [-center_box, center_box]^d
    centers = rng.uniform(-center_box, center_box, size=(n_classes, n_features))

    # 2) Optionally scale distances by class_sep
    centers *= class_sep

    # 3) Draw y ~ Categorical(weights)
    if weights is None:
        weights = np.ones(n_classes) / n_classes
    y = rng.choice(n_classes, size=n_samples, p=weights)

    # 4) Draw each X[i] ~ N(center_{y[i]}, cluster_std^2 I)
    X = centers[y] + rng.randn(n_samples, n_features) * cluster_std

    return X, y

if __name__=='__main__':

    X_all, y_all = generate_synthetic_data(n_samples=1000, n_features=2, n_classes=5)
    dataset = HistoricalDataset(X_all, y_all, pre_change_label=0)

    # 3. Instantiate any sklearn model
    rf = RandomForestClassifier(n_estimators=100)

    # 4. Train
    clf = train_sklearn_classifier(dataset, rf)

    # 5. Plug into your sequential test
    from sequential_test import SequentialHypothesisTest
    test = SequentialHypothesisTest(clf,
                                    n_classes=dataset.num_classes,
                                    alpha=0.01,
                                    M=200)
