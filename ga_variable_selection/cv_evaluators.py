"""
Fitness Evaluators for Genetic Algorithm Variable Selection
==========================================================

Implementations of various fitness functions for different problem types:
- PLS Regression (GAPLSR)
- LDA Classification (GALDA)
- FDA Classification (GAFDA)
- Mahalanobis Distance (GADIST, GAMAHAL)
- Distance-based similarity (GADISTA)

Each evaluator uses cross-validation to prevent overfitting.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union
import warnings


class FitnessEvaluator(ABC):
    """
    Abstract base class for fitness evaluators.

    All evaluators implement cross-validation and return a fitness
    score where higher values indicate better variable subsets.

    Parameters
    ----------
    config : dict
        Configuration parameters:
        - cv_method: 'kfold' or 'loocv' (default: 'kfold')
        - cv_groups: Number of folds for k-fold CV (default: 5)
        - metric: Metric type (depends on evaluator)
        - random_state: Random seed for reproducibility (default: None)
    """

    def __init__(self, config: dict):
        self.config = {
            'cv_method': 'kfold',
            'cv_groups': 5,
            'random_state': None,
            **config
        }

    @abstractmethod
    def evaluate(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        selected_indices: np.ndarray
    ) -> float:
        """
        Evaluate fitness of selected variables.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (n_samples, n_features)
        y : np.ndarray, optional
            Target variable (for supervised methods)
        selected_indices : np.ndarray
            Indices of selected features

        Returns
        -------
        fitness : float
            Fitness score (0-100 scale, higher is better)
        """
        pass

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_fn: callable,
        metric_fn: callable
    ) -> float:
        """
        Perform cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable
        model_fn : callable
            Function that fits model: model_fn(X_train, y_train) -> model
        metric_fn : callable
            Function that evaluates model: metric_fn(model, X_test, y_test) -> score

        Returns
        -------
        mean_score : float
            Mean cross-validation score
        """
        from sklearn.model_selection import KFold, LeaveOneOut

        n_samples = len(X)

        if self.config['cv_method'] == 'loocv':
            cv = LeaveOneOut()
        else:
            cv = KFold(
                n_splits=min(self.config['cv_groups'], n_samples),
                shuffle=True,
                random_state=self.config['random_state']
            )

        scores = []

        try:
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Fit model
                model = model_fn(X_train, y_train)

                # Evaluate
                score = metric_fn(model, X_test, y_test)
                scores.append(score)

            return np.mean(scores)

        except Exception as e:
            warnings.warn(f"Cross-validation failed: {str(e)}", RuntimeWarning)
            return 0.0


class PLSEvaluator(FitnessEvaluator):
    """
    PLS Regression fitness evaluator (GAPLSR).

    Evaluates variable subsets based on cross-validated R² from
    Partial Least Squares regression.

    Parameters
    ----------
    config : dict
        Configuration with additional parameters:
        - n_components: Number of PLS components (default: None, uses min(n_vars, n_samples-1))
        - max_components: Maximum components (default: 10)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.config.setdefault('n_components', None)
        self.config.setdefault('max_components', 10)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selected_indices: np.ndarray
    ) -> float:
        """
        Evaluate PLS regression fitness.

        Returns R² score multiplied by 100 (0-100 scale).
        """
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.metrics import r2_score

        if len(selected_indices) == 0:
            return 0.0

        n_samples, n_features = X.shape

        # Determine number of components
        if self.config['n_components'] is None:
            n_components = min(
                n_features,
                n_samples - 1,
                self.config['max_components']
            )
        else:
            n_components = min(
                self.config['n_components'],
                n_features,
                n_samples - 1
            )

        if n_components < 1:
            return 0.0

        def model_fn(X_train, y_train):
            """Fit PLS model."""
            model = PLSRegression(
                n_components=n_components,
                scale=True,
                max_iter=500,
                tol=1e-06
            )
            model.fit(X_train, y_train)
            return model

        def metric_fn(model, X_test, y_test):
            """Compute R² score."""
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            # Ensure non-negative (R² can be negative for bad models)
            return max(0, r2)

        # Cross-validate
        mean_r2 = self.cross_validate(X, y, model_fn, metric_fn)

        # Return on 0-100 scale
        return mean_r2 * 100


class LDAEvaluator(FitnessEvaluator):
    """
    Linear Discriminant Analysis fitness evaluator (GALDA).

    Evaluates variable subsets based on cross-validated classification
    accuracy using LDA.

    Parameters
    ----------
    config : dict
        Configuration (uses base class parameters)
    """

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selected_indices: np.ndarray
    ) -> float:
        """
        Evaluate LDA classification fitness.

        Returns accuracy percentage (0-100 scale).
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedKFold

        if len(selected_indices) == 0:
            return 0.0

        # Check classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        if n_classes < 2:
            warnings.warn("Need at least 2 classes for LDA", RuntimeWarning)
            return 0.0

        n_samples, n_features = X.shape

        # Check if enough samples per class
        class_counts = np.bincount(y.astype(int))
        if np.min(class_counts) < 2:
            warnings.warn("Need at least 2 samples per class", RuntimeWarning)
            return 0.0

        def model_fn(X_train, y_train):
            """Fit LDA model."""
            # Use SVD solver for numerical stability
            model = LinearDiscriminantAnalysis(
                solver='svd',
                store_covariance=False
            )
            try:
                model.fit(X_train, y_train)
                return model
            except:
                # Fallback to least squares solver
                model = LinearDiscriminantAnalysis(
                    solver='lsqr',
                    shrinkage='auto'
                )
                model.fit(X_train, y_train)
                return model

        def metric_fn(model, X_test, y_test):
            """Compute accuracy."""
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)

        # Use stratified CV to maintain class balance
        cv_groups = min(self.config['cv_groups'], np.min(class_counts))

        scores = []
        cv = StratifiedKFold(
            n_splits=cv_groups,
            shuffle=True,
            random_state=self.config['random_state']
        )

        try:
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = model_fn(X_train, y_train)
                score = metric_fn(model, X_test, y_test)
                scores.append(score)

            mean_accuracy = np.mean(scores)
            return mean_accuracy * 100

        except Exception as e:
            warnings.warn(f"LDA evaluation failed: {str(e)}", RuntimeWarning)
            return 0.0


class MahalanobisEvaluator(FitnessEvaluator):
    """
    Mahalanobis Distance fitness evaluator for fault detection (GADIST, GAMAHAL).

    Evaluates how well selected variables discriminate a faulty/anomalous
    point from normal samples using Mahalanobis distance.

    The dataset should contain normal samples and ONE faulty sample
    (typically the last row).

    Parameters
    ----------
    config : dict
        Configuration with additional parameters:
        - faulty_index: Index of faulty sample (default: -1, last sample)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.config.setdefault('faulty_index', -1)

    def evaluate(
        self,
        X: np.ndarray,
        selected_indices: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> float:
        """
        Evaluate Mahalanobis distance fitness.

        Parameters
        ----------
        X : np.ndarray
            Full dataset including faulty sample
        selected_indices : np.ndarray
            Selected variable indices
        y : np.ndarray, optional
            Not used (for API compatibility)

        Returns
        -------
        fitness : float
            Ratio of Mahalanobis distance to Euclidean distance
            (higher means better discrimination)
        """
        from scipy.spatial.distance import mahalanobis, euclidean
        from scipy.linalg import pinv

        if len(selected_indices) == 0:
            return 0.0

        n_samples = len(X)
        faulty_idx = self.config['faulty_index']

        # Separate normal and faulty samples
        if faulty_idx == -1:
            X_normal = X[:-1]
            x_faulty = X[-1]
        else:
            normal_mask = np.ones(n_samples, dtype=bool)
            normal_mask[faulty_idx] = False
            X_normal = X[normal_mask]
            x_faulty = X[faulty_idx]

        if len(X_normal) < 2:
            return 0.0

        # Standardize based on normal samples
        mean_normal = np.mean(X_normal, axis=0)
        std_normal = np.std(X_normal, axis=0)
        std_normal[std_normal == 0] = 1  # Avoid division by zero

        X_normal_std = (X_normal - mean_normal) / std_normal
        x_faulty_std = (x_faulty - mean_normal) / std_normal

        try:
            # Covariance matrix of normal samples
            cov_matrix = np.cov(X_normal_std.T)

            # Handle singular covariance
            if len(selected_indices) == 1:
                # Univariate case: use variance
                variance = cov_matrix if np.isscalar(cov_matrix) else cov_matrix[0, 0]
                if variance <= 0:
                    return 0.0
                inv_cov = 1.0 / variance
                mahal_dist = np.sqrt(np.abs((x_faulty_std ** 2) * inv_cov))
            else:
                # Multivariate case
                try:
                    inv_cov = np.linalg.inv(cov_matrix)
                except np.linalg.LinAlgError:
                    # Use pseudo-inverse for singular matrices
                    inv_cov = pinv(cov_matrix)

                # Mahalanobis distance
                diff = x_faulty_std
                mahal_dist = np.sqrt(np.abs(diff @ inv_cov @ diff.T))

            # Euclidean distance to nearest normal neighbor
            eucl_distances = np.sqrt(np.sum((X_normal_std - x_faulty_std) ** 2, axis=1))
            min_eucl_dist = np.min(eucl_distances)

            if min_eucl_dist == 0:
                return 0.0

            # Fitness: ratio of Mahalanobis to Euclidean distance
            fitness = mahal_dist / min_eucl_dist

            # Return on reasonable scale (0-100)
            # Mahal/Eucl ratio typically ranges 0-10, scale to 0-100
            return min(fitness * 10, 100)

        except Exception as e:
            warnings.warn(f"Mahalanobis evaluation failed: {str(e)}", RuntimeWarning)
            return 0.0


class DistanceEvaluator(FitnessEvaluator):
    """
    Distance-based fitness evaluator (GADISTA).

    Evaluates variable subsets based on class separation using
    inter-class vs intra-class distance ratio.

    Parameters
    ----------
    config : dict
        Configuration (uses base class parameters)
    """

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selected_indices: np.ndarray
    ) -> float:
        """
        Evaluate distance-based fitness.

        Returns ratio of inter-class to intra-class distances.
        """
        if len(selected_indices) == 0:
            return 0.0

        # Check classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        if n_classes < 2:
            return 0.0

        try:
            # Compute class centroids
            centroids = []
            intra_class_dists = []

            for cls in unique_classes:
                X_cls = X[y == cls]

                if len(X_cls) == 0:
                    continue

                # Centroid
                centroid = np.mean(X_cls, axis=0)
                centroids.append(centroid)

                # Intra-class distance (average distance to centroid)
                if len(X_cls) > 1:
                    dists = np.sqrt(np.sum((X_cls - centroid) ** 2, axis=1))
                    intra_class_dists.append(np.mean(dists))
                else:
                    intra_class_dists.append(0.0)

            if len(centroids) < 2:
                return 0.0

            # Inter-class distance (average pairwise centroid distance)
            inter_dists = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dist = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
                    inter_dists.append(dist)

            mean_inter = np.mean(inter_dists)
            mean_intra = np.mean(intra_class_dists)

            if mean_intra == 0:
                # Perfect within-class clustering
                return 100.0

            # Fitness: inter-class / intra-class ratio
            fitness = mean_inter / mean_intra

            # Scale to 0-100 range (ratio typically 0-10)
            return min(fitness * 10, 100)

        except Exception as e:
            warnings.warn(f"Distance evaluation failed: {str(e)}", RuntimeWarning)
            return 0.0


class FDAEvaluator(LDAEvaluator):
    """
    Fisher Discriminant Analysis fitness evaluator (GAFDA).

    Similar to LDA but uses Fisher's criterion (ratio of between-class
    to within-class variance) as the fitness metric instead of accuracy.

    Parameters
    ----------
    config : dict
        Configuration (uses base class parameters)
    """

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selected_indices: np.ndarray
    ) -> float:
        """
        Evaluate FDA fitness using Fisher criterion.

        Returns Fisher ratio scaled to 0-100.
        """
        if len(selected_indices) == 0:
            return 0.0

        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        if n_classes < 2:
            return 0.0

        try:
            # Compute overall mean
            overall_mean = np.mean(X, axis=0)

            # Compute between-class and within-class scatter
            between_scatter = np.zeros((X.shape[1], X.shape[1]))
            within_scatter = np.zeros((X.shape[1], X.shape[1]))

            for cls in unique_classes:
                X_cls = X[y == cls]
                n_cls = len(X_cls)

                if n_cls == 0:
                    continue

                # Class mean
                class_mean = np.mean(X_cls, axis=0)

                # Between-class scatter
                mean_diff = (class_mean - overall_mean).reshape(-1, 1)
                between_scatter += n_cls * (mean_diff @ mean_diff.T)

                # Within-class scatter
                for sample in X_cls:
                    sample_diff = (sample - class_mean).reshape(-1, 1)
                    within_scatter += sample_diff @ sample_diff.T

            # Fisher criterion: trace(between) / trace(within)
            trace_between = np.trace(between_scatter)
            trace_within = np.trace(within_scatter)

            if trace_within == 0:
                return 100.0

            fisher_ratio = trace_between / trace_within

            # Scale to 0-100 (ratio typically 0-10)
            return min(fisher_ratio * 10, 100)

        except Exception as e:
            warnings.warn(f"FDA evaluation failed: {str(e)}", RuntimeWarning)
            return 0.0


# Factory function for easy evaluator creation
def create_evaluator(problem_type: str, config: dict) -> FitnessEvaluator:
    """
    Factory function to create appropriate evaluator.

    Parameters
    ----------
    problem_type : str
        One of: 'pls', 'lda', 'fda', 'mahalanobis', 'distance'
    config : dict
        Configuration parameters

    Returns
    -------
    evaluator : FitnessEvaluator
        Initialized evaluator instance
    """
    evaluator_map = {
        'pls': PLSEvaluator,
        'lda': LDAEvaluator,
        'fda': FDAEvaluator,
        'mahalanobis': MahalanobisEvaluator,
        'distance': DistanceEvaluator
    }

    if problem_type.lower() not in evaluator_map:
        raise ValueError(
            f"Unknown problem type: {problem_type}. "
            f"Choose from: {list(evaluator_map.keys())}"
        )

    return evaluator_map[problem_type.lower()](config)
