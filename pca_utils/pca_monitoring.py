"""
PCA Process Monitoring Module

This module provides comprehensive PCA-based process monitoring capabilities including:
- Training PCA models from normal operating data
- Calculating T² (Hotelling) and Q (SPE) statistics
- Setting multiple control limits (97.5%, 99.5%, 99.95%)
- Fault detection on new process data
- Contribution analysis for fault diagnosis
- Save/load functionality for deployed models

References
----------
.. [1] Jackson, J.E. (1991). A User's Guide to Principal Components
.. [2] Nomikos & MacGregor (1995). Multivariate SPC charts for monitoring batch processes
.. [3] Kourti & MacGregor (1995). Process analysis, monitoring and diagnosis using multivariate projection methods
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional, List
from scipy.stats import f, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PCAMonitor:
    """
    PCA-based Statistical Process Monitoring

    This class implements PCA-based process monitoring for multivariate statistical
    process control (MSPC). It trains on normal operating condition (NOC) data and
    detects faults in new process data using T² and Q statistics.

    Parameters
    ----------
    n_components : int or float, optional
        Number of principal components to retain.
        If int: use that many components
        If float (0 < n_components < 1): select components to explain that fraction of variance
        Default is None (uses all components)
    scaling : str, optional
        Scaling method for data preprocessing.
        Options: 'auto' (standardization), 'pareto', 'none'
        Default is 'auto'
    alpha_levels : list of float, optional
        Confidence levels for control limits (e.g., [0.975, 0.995, 0.9995])
        Default is [0.975, 0.995, 0.9995] for 97.5%, 99.5%, 99.95%

    Attributes
    ----------
    pca_model_ : PCA
        Fitted scikit-learn PCA model
    scaler_ : StandardScaler
        Fitted data scaler
    t2_limits_ : dict
        T² control limits for each alpha level
    q_limits_ : dict
        Q control limits for each alpha level
    explained_variance_ : array
        Variance explained by each component
    loadings_ : array
        PCA loadings matrix (n_features, n_components)
    n_samples_train_ : int
        Number of training samples
    feature_names_ : list
        Names of input features

    Examples
    --------
    >>> # Train monitoring model
    >>> monitor = PCAMonitor(n_components=5)
    >>> monitor.fit(X_train)
    >>>
    >>> # Test new data
    >>> results = monitor.predict(X_test)
    >>> faults = results['faults']
    >>> print(f"Detected {faults.sum()} faulty samples")
    >>>
    >>> # Save model for deployment
    >>> monitor.save('pca_monitor_model.pkl')
    >>>
    >>> # Load and use
    >>> monitor_loaded = PCAMonitor.load('pca_monitor_model.pkl')
    >>> new_results = monitor_loaded.predict(X_new)
    """

    def __init__(
        self,
        n_components: Union[int, float, None] = None,
        scaling: str = 'auto',
        alpha_levels: List[float] = None
    ):
        self.n_components = n_components
        self.scaling = scaling
        self.alpha_levels = alpha_levels or [0.975, 0.995, 0.9995]

        # Model components (initialized during fit)
        self.pca_model_ = None
        self.scaler_ = None
        self.t2_limits_ = {}
        self.q_limits_ = {}
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.loadings_ = None
        self.mean_ = None
        self.std_ = None
        self.n_samples_train_ = None
        self.n_features_ = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], feature_names: List[str] = None):
        """
        Fit PCA monitoring model on normal operating condition (NOC) data.

        This method trains the PCA model and calculates control limits based on
        the training data statistics.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data from normal operating conditions
        feature_names : list of str, optional
            Names of features (variables). If None and X is DataFrame, uses column names

        Returns
        -------
        self : PCAMonitor
            Fitted monitoring model
        """
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = feature_names or list(X.columns)
            X_array = X.values
        else:
            X_array = np.asarray(X)
            self.feature_names_ = feature_names or [f'Var{i+1}' for i in range(X_array.shape[1])]

        self.n_samples_train_, self.n_features_ = X_array.shape

        # Scale data
        X_scaled = self._scale_data(X_array, fit=True)

        # Fit PCA model
        if self.n_components is None:
            n_comp = min(self.n_samples_train_, self.n_features_)
        elif isinstance(self.n_components, float):
            # Determine n_components from variance explained
            n_comp = self.n_components  # PCA will handle this
        else:
            n_comp = min(self.n_components, self.n_samples_train_, self.n_features_)

        self.pca_model_ = PCA(n_components=n_comp)
        scores_train = self.pca_model_.fit_transform(X_scaled)

        # Store PCA parameters
        self.explained_variance_ = self.pca_model_.explained_variance_
        self.explained_variance_ratio_ = self.pca_model_.explained_variance_ratio_
        self.loadings_ = self.pca_model_.components_.T  # (n_features, n_components)

        # Calculate T² control limits for each alpha level
        n_comp_actual = self.pca_model_.n_components_
        for alpha in self.alpha_levels:
            self.t2_limits_[alpha] = self._calculate_t2_limit(
                self.n_samples_train_, n_comp_actual, alpha
            )

        # Calculate training Q statistics to determine limits
        q_train = self._calculate_q_statistics(X_scaled, scores_train)

        # Calculate Q control limits for each alpha level
        for alpha in self.alpha_levels:
            self.q_limits_[alpha] = self._calculate_q_limit(q_train, alpha)

        self.is_fitted_ = True

        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_contributions: bool = True
    ) -> Dict[str, Any]:
        """
        Test new data for faults using trained monitoring model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New process data to monitor
        return_contributions : bool, optional
            If True, calculate variable contributions for fault diagnosis
            Default is True

        Returns
        -------
        results : dict
            Dictionary containing:
            - 't2': T² statistic for each sample
            - 'q': Q statistic for each sample
            - 't2_limits': dict of T² control limits
            - 'q_limits': dict of Q control limits
            - 'faults': boolean array indicating fault detection
            - 'fault_type': classification of each fault ('none', 't2', 'q', 'both')
            - 'scores': PCA scores
            - 'X_scaled': scaled data
            - 'contributions_t2': T² contributions (if return_contributions=True)
            - 'contributions_q': Q contributions (if return_contributions=True)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)

        n_samples = X_array.shape[0]

        # Scale data using training parameters
        X_scaled = self._scale_data(X_array, fit=False)

        # Project to PCA space
        scores = self.pca_model_.transform(X_scaled)

        # Calculate T² statistics
        t2 = self._calculate_t2_statistics(scores)

        # Calculate Q statistics
        q = self._calculate_q_statistics(X_scaled, scores)

        # Detect faults using most conservative limit (usually 97.5%)
        primary_alpha = min(self.alpha_levels)
        t2_faults = t2 > self.t2_limits_[primary_alpha]
        q_faults = q > self.q_limits_[primary_alpha]

        # Classify fault types
        fault_type = np.array(['none'] * n_samples, dtype=object)
        fault_type[t2_faults & ~q_faults] = 't2'
        fault_type[~t2_faults & q_faults] = 'q'
        fault_type[t2_faults & q_faults] = 'both'

        faults = t2_faults | q_faults

        results = {
            't2': t2,
            'q': q,
            't2_limits': self.t2_limits_,
            'q_limits': self.q_limits_,
            'faults': faults,
            'fault_type': fault_type,
            'scores': scores,
            'X_scaled': X_scaled
        }

        # Calculate contributions if requested
        if return_contributions:
            contrib_t2, contrib_q = self._calculate_contributions(X_scaled, scores)
            results['contributions_t2'] = contrib_t2
            results['contributions_q'] = contrib_q

        return results

    def plot_monitoring_chart(
        self,
        results: Dict[str, Any],
        sample_labels: Optional[List[str]] = None,
        title: str = "PCA Monitoring Chart"
    ) -> go.Figure:
        """
        Create interactive monitoring chart showing T² and Q statistics.

        Parameters
        ----------
        results : dict
            Results from predict() method
        sample_labels : list of str, optional
            Labels for each sample (e.g., timestamps, sample IDs)
        title : str, optional
            Chart title

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive monitoring chart
        """
        t2 = results['t2']
        q = results['q']
        n_samples = len(t2)

        if sample_labels is None:
            sample_labels = [f"Sample {i+1}" for i in range(n_samples)]

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Hotelling T² Statistic', 'Q Statistic (SPE)'),
            vertical_spacing=0.12
        )

        # T² chart
        fig.add_trace(
            go.Scatter(
                x=list(range(n_samples)),
                y=t2,
                mode='lines+markers',
                name='T²',
                line=dict(color='steelblue', width=2),
                marker=dict(size=6),
                text=sample_labels,
                hovertemplate='<b>%{text}</b><br>T²: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # T² control limits
        colors = ['green', 'orange', 'red']
        for idx, (alpha, color) in enumerate(zip(sorted(self.alpha_levels), colors)):
            limit = self.t2_limits_[alpha]
            confidence_pct = alpha * 100
            fig.add_hline(
                y=limit, line_dash="dash", line_color=color,
                annotation_text=f"{confidence_pct:.1f}% CL",
                row=1, col=1
            )

        # Q chart
        fig.add_trace(
            go.Scatter(
                x=list(range(n_samples)),
                y=q,
                mode='lines+markers',
                name='Q',
                line=dict(color='coral', width=2),
                marker=dict(size=6),
                text=sample_labels,
                hovertemplate='<b>%{text}</b><br>Q: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Q control limits
        for idx, (alpha, color) in enumerate(zip(sorted(self.alpha_levels), colors)):
            limit = self.q_limits_[alpha]
            confidence_pct = alpha * 100
            fig.add_hline(
                y=limit, line_dash="dash", line_color=color,
                annotation_text=f"{confidence_pct:.1f}% CL",
                row=2, col=1
            )

        # Update layout
        fig.update_xaxes(title_text="Sample Number", row=2, col=1)
        fig.update_yaxes(title_text="T² Statistic", row=1, col=1)
        fig.update_yaxes(title_text="Q Statistic", row=2, col=1)

        fig.update_layout(
            title=title,
            height=700,
            showlegend=False,
            hovermode='x unified'
        )

        return fig

    def plot_contribution_chart(
        self,
        results: Dict[str, Any],
        sample_idx: int,
        statistic: str = 'q',
        top_n: int = 15
    ) -> go.Figure:
        """
        Create contribution plot for fault diagnosis.

        Shows which variables contributed most to the fault detection for a specific sample.

        Parameters
        ----------
        results : dict
            Results from predict() method (must include contributions)
        sample_idx : int
            Index of the sample to analyze
        statistic : str, optional
            Which statistic to show contributions for: 't2' or 'q'
            Default is 'q'
        top_n : int, optional
            Number of top contributing variables to display
            Default is 15

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive contribution chart
        """
        if statistic.lower() == 't2':
            if 'contributions_t2' not in results:
                raise ValueError("T² contributions not available. Set return_contributions=True in predict()")
            contributions = results['contributions_t2'][sample_idx]
            title_stat = 'T²'
        elif statistic.lower() == 'q':
            if 'contributions_q' not in results:
                raise ValueError("Q contributions not available. Set return_contributions=True in predict()")
            contributions = results['contributions_q'][sample_idx]
            title_stat = 'Q'
        else:
            raise ValueError("statistic must be 't2' or 'q'")

        # Get top contributors
        abs_contributions = np.abs(contributions)
        top_indices = np.argsort(abs_contributions)[-top_n:][::-1]

        top_vars = [self.feature_names_[i] for i in top_indices]
        top_contribs = contributions[top_indices]

        # Create bar chart
        colors = ['red' if c > 0 else 'steelblue' for c in top_contribs]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=top_contribs,
            y=top_vars,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{c:.2f}' for c in top_contribs],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Contribution: %{x:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'{title_stat} Contribution Plot - Sample {sample_idx + 1}',
            xaxis_title=f'{title_stat} Contribution',
            yaxis_title='Variable',
            height=max(400, top_n * 25),
            showlegend=False
        )

        fig.add_vline(x=0, line_width=1, line_color="black")

        return fig

    def get_fault_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate summary statistics for detected faults.

        Parameters
        ----------
        results : dict
            Results from predict() method

        Returns
        -------
        summary : pd.DataFrame
            Summary table with fault information for each sample
        """
        n_samples = len(results['t2'])

        summary_data = {
            'Sample': list(range(1, n_samples + 1)),
            'T2_Statistic': results['t2'],
            'Q_Statistic': results['q'],
            'Fault_Detected': results['faults'],
            'Fault_Type': results['fault_type']
        }

        # Add limit exceedance info for primary alpha level
        primary_alpha = min(self.alpha_levels)
        summary_data[f'T2_Exceeds_{primary_alpha*100:.1f}%'] = results['t2'] > results['t2_limits'][primary_alpha]
        summary_data[f'Q_Exceeds_{primary_alpha*100:.1f}%'] = results['q'] > results['q_limits'][primary_alpha]

        summary_df = pd.DataFrame(summary_data)

        return summary_df

    def save(self, filepath: Union[str, Path]):
        """
        Save trained monitoring model to file.

        Parameters
        ----------
        filepath : str or Path
            Path where model should be saved (pickle format)
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted model. Call fit() first.")

        filepath = Path(filepath)

        model_data = {
            'n_components': self.n_components,
            'scaling': self.scaling,
            'alpha_levels': self.alpha_levels,
            'pca_model': self.pca_model_,
            'scaler': self.scaler_,
            't2_limits': self.t2_limits_,
            'q_limits': self.q_limits_,
            'explained_variance': self.explained_variance_,
            'explained_variance_ratio': self.explained_variance_ratio_,
            'loadings': self.loadings_,
            'mean': self.mean_,
            'std': self.std_,
            'n_samples_train': self.n_samples_train_,
            'n_features': self.n_features_,
            'feature_names': self.feature_names_
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'PCAMonitor':
        """
        Load trained monitoring model from file.

        Parameters
        ----------
        filepath : str or Path
            Path to saved model file

        Returns
        -------
        monitor : PCAMonitor
            Loaded monitoring model ready for prediction
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create new instance
        monitor = cls(
            n_components=model_data['n_components'],
            scaling=model_data['scaling'],
            alpha_levels=model_data['alpha_levels']
        )

        # Restore fitted parameters
        monitor.pca_model_ = model_data['pca_model']
        monitor.scaler_ = model_data['scaler']
        monitor.t2_limits_ = model_data['t2_limits']
        monitor.q_limits_ = model_data['q_limits']
        monitor.explained_variance_ = model_data['explained_variance']
        monitor.explained_variance_ratio_ = model_data['explained_variance_ratio']
        monitor.loadings_ = model_data['loadings']
        monitor.mean_ = model_data['mean']
        monitor.std_ = model_data['std']
        monitor.n_samples_train_ = model_data['n_samples_train']
        monitor.n_features_ = model_data['n_features']
        monitor.feature_names_ = model_data['feature_names']
        monitor.is_fitted_ = True

        print(f"Model loaded from {filepath}")

        return monitor

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of fitted model parameters.

        Returns
        -------
        summary : dict
            Dictionary with model information
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")

        summary = {
            'n_components': self.pca_model_.n_components_,
            'n_features': self.n_features_,
            'n_samples_train': self.n_samples_train_,
            'variance_explained': np.sum(self.explained_variance_ratio_),
            'variance_per_pc': self.explained_variance_ratio_.tolist(),
            'scaling': self.scaling,
            't2_limits': self.t2_limits_,
            'q_limits': self.q_limits_
        }

        return summary

    # Private helper methods

    def _scale_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale data according to specified method."""
        if self.scaling == 'none':
            if fit:
                self.mean_ = np.zeros(X.shape[1])
                self.std_ = np.ones(X.shape[1])
            return X

        elif self.scaling == 'auto':
            # Mean centering and standardization
            if fit:
                self.mean_ = np.mean(X, axis=0)
                self.std_ = np.std(X, axis=0)
                self.std_[self.std_ == 0] = 1  # Avoid division by zero
            X_scaled = (X - self.mean_) / self.std_

        elif self.scaling == 'pareto':
            # Pareto scaling (mean centering and divide by sqrt of std)
            if fit:
                self.mean_ = np.mean(X, axis=0)
                self.std_ = np.sqrt(np.std(X, axis=0))
                self.std_[self.std_ == 0] = 1
            X_scaled = (X - self.mean_) / self.std_

        else:
            raise ValueError(f"Unknown scaling method: {self.scaling}")

        return X_scaled

    def _calculate_t2_statistics(self, scores: np.ndarray) -> np.ndarray:
        """Calculate T² statistic for given scores."""
        # T² = sum((score² / eigenvalue))
        t2 = np.sum((scores ** 2) / self.explained_variance_, axis=1)
        return t2

    def _calculate_t2_limit(self, n: int, a: int, alpha: float) -> float:
        """
        Calculate T² control limit using F-distribution.

        T² ~ [(n-1)*a / (n-a)] * F(a, n-a, alpha)
        """
        if n <= a:
            return 1e10  # Very large limit if not enough samples

        f_value = f.ppf(alpha, a, n - a)
        t2_limit = ((n - 1) * a / (n - a)) * f_value

        return t2_limit

    def _calculate_q_statistics(self, X_scaled: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Calculate Q (SPE) statistic."""
        # Reconstruct data
        X_reconstructed = scores @ self.loadings_.T

        # Calculate residuals
        residuals = X_scaled - X_reconstructed

        # Q statistic = sum of squared residuals
        q = np.sum(residuals ** 2, axis=1)

        return q

    def _calculate_q_limit(self, q_train: np.ndarray, alpha: float) -> float:
        """
        Calculate Q control limit using chi-square approximation.

        Simplified Jackson-Mudholkar method.
        """
        q_mean = np.mean(q_train)
        q_var = np.var(q_train)

        if q_var == 0 or q_mean == 0:
            return np.percentile(q_train, alpha * 100)

        # Chi-square approximation parameters
        g = q_var / (2 * q_mean)
        h = (2 * q_mean ** 2) / q_var

        q_limit = g * chi2.ppf(alpha, h)

        return q_limit

    def _calculate_contributions(
        self,
        X_scaled: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate variable contributions to T² and Q statistics.

        Returns
        -------
        contrib_t2 : array, shape (n_samples, n_features)
            T² contributions
        contrib_q : array, shape (n_samples, n_features)
            Q contributions
        """
        n_samples, n_features = X_scaled.shape

        # T² contributions
        # Contribution of variable j to T² = sum over PCs of (loading_j * score_i / eigenvalue)
        contrib_t2 = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            for j in range(n_features):
                contrib = 0
                for k in range(self.pca_model_.n_components_):
                    contrib += (self.loadings_[j, k] * scores[i, k] ** 2) / self.explained_variance_[k]
                contrib_t2[i, j] = contrib

        # Q contributions
        # Contribution of variable j to Q = (residual_j)²
        X_reconstructed = scores @ self.loadings_.T
        residuals = X_scaled - X_reconstructed
        contrib_q = residuals ** 2

        return contrib_t2, contrib_q


def plot_combined_monitoring_chart(
    results: Dict[str, Any],
    t2_limits: Dict[float, float],
    q_limits: Dict[float, float],
    sample_labels: Optional[List[str]] = None,
    title: str = "PCA Monitoring - T² vs Q Chart"
) -> go.Figure:
    """
    Create combined T² vs Q scatter plot for process monitoring.

    This plot shows the relationship between T² and Q statistics,
    with control limit boundaries marked. Useful for visualizing
    different types of faults.

    Parameters
    ----------
    results : dict
        Results from PCAMonitor.predict()
    t2_limits : dict
        T² control limits
    q_limits : dict
        Q control limits
    sample_labels : list of str, optional
        Labels for samples
    title : str, optional
        Chart title

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive scatter plot
    """
    t2 = results['t2']
    q = results['q']
    fault_type = results['fault_type']
    n_samples = len(t2)

    if sample_labels is None:
        sample_labels = [f"Sample {i+1}" for i in range(n_samples)]

    # Color mapping for fault types
    color_map = {
        'none': 'green',
        't2': 'orange',
        'q': 'blue',
        'both': 'red'
    }
    colors = [color_map[ft] for ft in fault_type]

    # Create scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t2,
        y=q,
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            line=dict(width=1, color='white')
        ),
        text=sample_labels,
        customdata=fault_type,
        hovertemplate='<b>%{text}</b><br>T²: %{x:.2f}<br>Q: %{y:.2f}<br>Type: %{customdata}<extra></extra>'
    ))

    # Add control limit lines
    primary_alpha = min(t2_limits.keys())
    t2_lim = t2_limits[primary_alpha]
    q_lim = q_limits[primary_alpha]

    fig.add_vline(x=t2_lim, line_dash="dash", line_color="red",
                  annotation_text=f"T² limit ({primary_alpha*100:.1f}%)")
    fig.add_hline(y=q_lim, line_dash="dash", line_color="red",
                  annotation_text=f"Q limit ({primary_alpha*100:.1f}%)")

    fig.update_layout(
        title=title,
        xaxis_title='T² Statistic',
        yaxis_title='Q Statistic',
        height=600,
        showlegend=False,
        hovermode='closest'
    )

    return fig
