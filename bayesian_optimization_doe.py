"""
Bayesian Optimization Designer for Design of Experiments
=========================================================

This module provides a complete Bayesian Optimization (BO) framework for:
- Suggesting new experimental points based on existing data
- Using Gaussian Process surrogate models
- Multiple acquisition functions (EI, LCB)
- Visualization of optimization landscapes

Author: ChemoMetric Solutions
License: MIT
"""

# IMPORTS
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional

# Import utilities
try:
    from bayesian_utils import (
        validate_bounds,
        compute_acquisition_ei,
        compute_acquisition_lcb,
        normalize_to_coded,
        denormalize_to_natural,
        parse_constraint_string
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    # Fallback: will use built-in implementations if needed


class BayesianOptimizationDesigner:
    """
    Bayesian Optimization Designer for experimental design.

    Uses Gaussian Process regression to model the relationship between
    factors and responses, then suggests new experimental points that
    maximize an acquisition function (Expected Improvement or Lower
    Confidence Bound).

    Attributes:
        data: DataFrame containing experimental observations
        numeric_features: List of numeric column names
        bounds_dict: Dictionary storing factor bounds and types
        constraints: List of constraint functions
        gp_model: Fitted Gaussian Process model
        suggested_points_df: DataFrame of suggested experimental points
        optimization_history: List of optimization runs
    """

    def __init__(self, data_df: pd.DataFrame, workspace_metadata: dict = None):
        """
        Initialize Bayesian Optimization Designer with workspace data.

        Args:
            data_df: DataFrame from st.session_state.current_data
            workspace_metadata: Optional metadata about workspace
        """
        self.data = data_df.copy()
        self.numeric_features = data_df.select_dtypes(include=[np.number]).columns.tolist()
        self.bounds_dict = {}
        self.constraints = []
        self.gp_model = None
        self.suggested_points_df = None
        self.optimization_history = []
        self.workspace_metadata = workspace_metadata or {}

        # Training data storage
        self.X_train = None
        self.y_train = None
        self.gp_score = None

    def validate_workspace_data(self) -> Tuple[bool, str]:
        """
        Check if data is suitable for Bayesian Optimization.

        Returns:
            Tuple of (is_valid, message)
        """
        if len(self.data) < 3:
            return False, "Need at least 3 observations for BO"

        if len(self.numeric_features) < 2:
            return False, "Need at least 2 numeric features (factors)"

        # Check for variance in data
        for col in self.numeric_features:
            if self.data[col].std() == 0:
                return False, f"Feature '{col}' has no variance"

        return True, "Data valid for Bayesian Optimization"

    def detect_experimental_factors(self) -> List[str]:
        """
        Auto-detect which columns are suitable experimental factors.

        Returns:
            List of factor names
        """
        factors = []
        for col in self.numeric_features:
            # At least 3 unique values
            if self.data[col].nunique() >= 3:
                col_range = self.data[col].max() - self.data[col].min()
                # Not constant
                if col_range > 0:
                    factors.append(col)
        return factors

    def set_factor_bounds(self, factor_name: str, lower: float, upper: float,
                         factor_type: str = "continuous", step: float = None):
        """
        Store bounds for each experimental factor.

        Args:
            factor_name: Name of the factor
            lower: Lower bound
            upper: Upper bound
            factor_type: "continuous" or "discrete"
            step: Step size for discrete factors

        Raises:
            ValueError: If bounds are invalid
        """
        if lower >= upper:
            raise ValueError(f"Bounds error for {factor_name}: lower >= upper ({lower} >= {upper})")

        self.bounds_dict[factor_name] = {
            'lower': float(lower),
            'upper': float(upper),
            'type': factor_type.lower(),
            'step': float(step) if step else None
        }

    def fit_gaussian_process_from_data(self, X_cols: List[str], y_col: str,
                                       kernel_type: str = "Matern52") -> bool:
        """
        Fit Gaussian Process surrogate model using workspace data.

        Args:
            X_cols: List of factor column names
            y_col: Response column name
            kernel_type: "Matern32", "Matern52", or "RBF"

        Returns:
            True if fitting successful

        Raises:
            ValueError: If insufficient data
            Exception: If GP fitting fails
        """
        try:
            # Extract and clean data
            X = self.data[X_cols].dropna().values
            y = self.data[y_col].dropna().values

            # Ensure matching lengths
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]

            if len(X) < 3:
                raise ValueError("Need at least 3 complete observations for GP fitting")

            # Select kernel
            if kernel_type == "Matern32":
                kernel = Matern(nu=1.5)
            elif kernel_type == "RBF":
                kernel = RBF()
            else:  # Matern52 (default)
                kernel = Matern(nu=2.5)

            # Fit Gaussian Process
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=42
            )
            self.gp_model.fit(X, y)

            # Store metadata
            self.gp_score = self.gp_model.score(X, y)
            self.X_train = X
            self.y_train = y
            self.X_cols = X_cols
            self.y_col = y_col

            return True

        except Exception as e:
            raise Exception(f"GP fitting failed: {str(e)}")

    def run_bayesian_optimization(self, batch_size: int = 5,
                                 acquisition: str = "EI",
                                 maximize: bool = True,
                                 jitter: float = 0.01) -> pd.DataFrame:
        """
        Execute Bayesian Optimization to suggest next experimental points.

        Args:
            batch_size: Number of points to suggest
            acquisition: "EI" (Expected Improvement) or "LCB" (Lower Confidence Bound)
            maximize: Whether to maximize (True) or minimize (False) target
            jitter: Exploration-exploitation trade-off parameter

        Returns:
            DataFrame with suggested experimental points

        Raises:
            ValueError: If GP model not fitted
        """
        if self.gp_model is None:
            raise ValueError("Must fit GP model first using fit_gaussian_process_from_data()")

        factor_names = list(self.bounds_dict.keys())

        if len(factor_names) == 0:
            raise ValueError("No factor bounds set. Use set_factor_bounds() first.")

        # Create domain bounds for optimization
        domain_bounds = np.array([
            [self.bounds_dict[f]['lower'], self.bounds_dict[f]['upper']]
            for f in factor_names
        ])

        # Define acquisition function
        def acq_func(X):
            """Compute acquisition function value."""
            if X.ndim == 1:
                X = X.reshape(1, -1)

            mu, sigma = self.gp_model.predict(X, return_std=True)
            sigma = np.maximum(sigma, 1e-8)  # Avoid division by zero

            if maximize:
                y_best = self.y_train.max()
                improvement = mu - y_best - jitter
            else:
                y_best = self.y_train.min()
                improvement = y_best - mu - jitter

            if acquisition == "EI":
                # Expected Improvement
                Z = improvement / sigma
                ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
                return -ei  # Negative because we minimize
            else:  # LCB
                lcb = mu - jitter * sigma
                return -lcb if maximize else lcb

        # Optimize acquisition function to find candidate points
        n_restart = min(batch_size * 2, 50)
        candidates_list = []

        for _ in range(batch_size):
            best_x = None
            best_acq = np.inf

            # Multi-start optimization
            for _ in range(n_restart):
                x0 = np.random.uniform(domain_bounds[:, 0], domain_bounds[:, 1])
                result = minimize(acq_func, x0, bounds=domain_bounds, method='L-BFGS-B')

                if result.fun < best_acq:
                    best_acq = result.fun
                    best_x = result.x

            if best_x is not None:
                candidates_list.append(best_x)

        # Create results dataframe
        candidates_array = np.array(candidates_list)
        mu_pred, sigma_pred = self.gp_model.predict(candidates_array, return_std=True)

        # Compute acquisition values for display
        y_best = self.y_train.max() if maximize else self.y_train.min()
        acq_vals = []
        for i in range(len(mu_pred)):
            if maximize:
                improvement = mu_pred[i] - y_best - jitter
            else:
                improvement = y_best - mu_pred[i] - jitter

            Z = improvement / sigma_pred[i]
            ei = improvement * norm.cdf(Z) + sigma_pred[i] * norm.pdf(Z)
            acq_vals.append(ei)

        acq_vals = np.array(acq_vals)

        # Build results dataframe
        results_data = {fname: candidates_array[:, i]
                       for i, fname in enumerate(factor_names)}
        results_data['Expected_Target'] = mu_pred
        results_data['Std_Dev'] = sigma_pred
        results_data['Acquisition_Value'] = acq_vals

        self.suggested_points_df = pd.DataFrame(results_data)

        # Store optimization history
        self.optimization_history.append({
            'timestamp': pd.Timestamp.now(),
            'batch_size': batch_size,
            'acquisition': acquisition,
            'maximize': maximize,
            'n_suggested': len(self.suggested_points_df)
        })

        # VALIDATION: Check if BO returned suspicious identical points
        unique_acq_scores = len(self.suggested_points_df['Acquisition_Value'].unique())
        unique_predictions = len(self.suggested_points_df['Expected_Target'].unique())

        if unique_acq_scores == 1 or unique_predictions == 1:
            # Flag: all points have identical acquisition/prediction values
            self.bo_warning = {
                'severity': 'WARNING',
                'message': 'BO returned identical suggested points',
                'causes': [
                    'Too few training samples for number of factors',
                    'GP overfitting (check R² value)',
                    'Search space too constrained',
                    'All candidate points in same region'
                ],
                'recommendations': [
                    'Use Plackett-Burman screening FIRST to reduce factor count',
                    'Increase training dataset size (need 3x factors minimum)',
                    'Increase jitter parameter for more exploration',
                    'Expand bounds for search space'
                ],
                'gp_r_squared': self.gp_score,
                'n_training_samples': len(self.X_train),
                'n_factors': len(self.bounds_dict)
            }
        else:
            self.bo_warning = None

        return self.suggested_points_df

    def run_bayesian_optimization_on_grid(self, candidate_grid: np.ndarray,
                                          batch_size: int = 5,
                                          acquisition: str = "EI",
                                          maximize: bool = True,
                                          jitter: float = 0.01) -> pd.DataFrame:
        """
        Execute Bayesian Optimization by ranking a pre-computed candidate grid.

        Instead of using gradient-based optimization, this method evaluates
        the acquisition function on ALL provided candidate points and returns
        the top batch_size candidates.

        Args:
            candidate_grid: Array of candidate points, shape (n_candidates, n_factors)
            batch_size: Number of best points to return
            acquisition: "EI" (Expected Improvement) or "LCB" (Lower Confidence Bound)
            maximize: Whether to maximize (True) or minimize (False) target
            jitter: Exploration-exploitation trade-off parameter

        Returns:
            DataFrame with top batch_size suggested experimental points

        Raises:
            ValueError: If GP model not fitted or candidate_grid has wrong shape
        """
        if self.gp_model is None:
            raise ValueError("Must fit GP model first using fit_gaussian_process_from_data()")

        factor_names = list(self.bounds_dict.keys())

        if len(factor_names) == 0:
            raise ValueError("No factor bounds set. Use set_factor_bounds() first.")

        # Validate candidate grid shape
        if candidate_grid.ndim != 2:
            raise ValueError(f"candidate_grid must be 2D, got shape {candidate_grid.shape}")

        if candidate_grid.shape[1] != len(factor_names):
            raise ValueError(
                f"candidate_grid has {candidate_grid.shape[1]} columns but "
                f"expected {len(factor_names)} factors"
            )

        n_candidates = len(candidate_grid)

        # Predict on all candidates
        mu_all, sigma_all = self.gp_model.predict(candidate_grid, return_std=True)
        sigma_all = np.maximum(sigma_all, 1e-8)  # Avoid division by zero

        # Compute acquisition function for all candidates
        y_best = self.y_train.max() if maximize else self.y_train.min()

        if acquisition == "EI":
            # Expected Improvement
            if maximize:
                improvement = mu_all - y_best - jitter
            else:
                improvement = y_best - mu_all - jitter

            Z = improvement / sigma_all
            acq_vals = improvement * norm.cdf(Z) + sigma_all * norm.pdf(Z)
            acq_vals = np.maximum(acq_vals, 0.0)  # Ensure non-negative

        else:  # LCB
            # Lower Confidence Bound
            if maximize:
                acq_vals = mu_all + jitter * sigma_all  # UCB for maximization
            else:
                acq_vals = mu_all - jitter * sigma_all  # LCB for minimization

        # Rank candidates by acquisition value
        if maximize or acquisition == "EI":
            # Higher acquisition is better
            top_indices = np.argsort(acq_vals)[::-1][:batch_size]
        else:
            # Lower acquisition is better (for LCB minimization)
            top_indices = np.argsort(acq_vals)[:batch_size]

        # Extract top candidates
        best_candidates = candidate_grid[top_indices]
        best_mu = mu_all[top_indices]
        best_sigma = sigma_all[top_indices]
        best_acq = acq_vals[top_indices]

        # Build results dataframe
        results_data = {fname: best_candidates[:, i]
                       for i, fname in enumerate(factor_names)}
        results_data['Expected_Target'] = best_mu
        results_data['Std_Dev'] = best_sigma
        results_data['Acquisition_Value'] = best_acq

        self.suggested_points_df = pd.DataFrame(results_data)

        # Store optimization history
        self.optimization_history.append({
            'timestamp': pd.Timestamp.now(),
            'batch_size': batch_size,
            'acquisition': acquisition,
            'maximize': maximize,
            'n_suggested': len(self.suggested_points_df),
            'method': 'grid_search',
            'n_candidates_evaluated': n_candidates
        })

        # VALIDATION: Check if BO returned suspicious identical points
        unique_acq_scores = len(self.suggested_points_df['Acquisition_Value'].unique())
        unique_predictions = len(self.suggested_points_df['Expected_Target'].unique())

        if unique_acq_scores == 1 or unique_predictions == 1:
            self.bo_warning = {
                'severity': 'WARNING',
                'message': 'BO returned identical suggested points from grid',
                'causes': [
                    'Too few training samples for number of factors',
                    'GP overfitting (check R² value)',
                    'Grid resolution too coarse',
                    'All grid points in same region'
                ],
                'recommendations': [
                    'Use Plackett-Burman screening FIRST to reduce factor count',
                    'Increase training dataset size (need 3x factors minimum)',
                    'Increase grid resolution (more discrete steps)',
                    'Expand bounds for search space'
                ],
                'gp_r_squared': self.gp_score,
                'n_training_samples': len(self.X_train),
                'n_factors': len(self.bounds_dict),
                'n_candidates_evaluated': n_candidates
            }
        else:
            self.bo_warning = None

        return self.suggested_points_df

    def visualize_acquisition_1d(self, factor_idx: int, resolution: int = 100) -> go.Figure:
        """
        Plot 1D acquisition landscape for a single factor.

        Args:
            factor_idx: Index of factor to visualize
            resolution: Number of points to evaluate

        Returns:
            Plotly Figure object
        """
        factor_names = list(self.bounds_dict.keys())
        factor_name = factor_names[factor_idx]

        bounds = self.bounds_dict[factor_name]
        x_values = np.linspace(bounds['lower'], bounds['upper'], resolution)

        # Fix other factors at mean
        mean_values = {fname: self.bounds_dict[fname]['lower'] +
                      (self.bounds_dict[fname]['upper'] - self.bounds_dict[fname]['lower']) / 2
                      for fname in factor_names}

        predictions = []
        stds = []
        for x_val in x_values:
            X_test = np.array([[mean_values[fname] if i != factor_idx else x_val
                              for i, fname in enumerate(factor_names)]])
            mu, std = self.gp_model.predict(X_test, return_std=True)
            predictions.append(mu[0])
            stds.append(std[0])

        predictions = np.array(predictions)
        stds = np.array(stds)

        # Create figure
        fig = go.Figure()

        # Add mean prediction
        fig.add_trace(go.Scatter(
            x=x_values, y=predictions,
            name='GP Mean',
            line=dict(color='green', width=2),
            mode='lines'
        ))

        # Add std band
        upper_band = predictions + stds
        lower_band = predictions - stds
        fig.add_trace(go.Scatter(
            x=list(x_values) + list(x_values[::-1]),
            y=list(upper_band) + list(lower_band[::-1]),
            fill='toself',
            name='±1 Std',
            fillcolor='rgba(255, 127, 14, 0.3)',
            line=dict(color='rgba(255, 127, 14, 0)'),
            hoverinfo='skip'
        ))

        # Add data points if available
        if self.X_train is not None:
            X_subset = self.X_train[:, factor_idx]
            fig.add_trace(go.Scatter(
                x=X_subset, y=self.y_train,
                name='Observed Data',
                mode='markers',
                marker=dict(color='blue', size=8)
            ))

        # Add suggested point (if available)
        if self.suggested_points_df is not None and factor_name in self.suggested_points_df.columns:
            suggested_x = self.suggested_points_df[factor_name].iloc[0]
            suggested_y = self.suggested_points_df['Expected_Target'].iloc[0]
            fig.add_vline(x=suggested_x, line_dash="dash", line_color="red",
                         annotation_text="Suggested")

        fig.update_layout(
            title=f"Acquisition Landscape: {factor_name}",
            xaxis_title=f"{factor_name} (natural units)",
            yaxis_title="Target Prediction",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )

        return fig

    def visualize_acquisition_2d(self, factor1_idx: int, factor2_idx: int,
                                plot_type: str = "heatmap") -> go.Figure:
        """
        Plot 2D acquisition landscape for two factors.

        Args:
            factor1_idx: Index of first factor
            factor2_idx: Index of second factor
            plot_type: "heatmap" or "surface"

        Returns:
            Plotly Figure object
        """
        factor_names = list(self.bounds_dict.keys())
        f1_name = factor_names[factor1_idx]
        f2_name = factor_names[factor2_idx]

        # Create grid
        f1_range = np.linspace(self.bounds_dict[f1_name]['lower'],
                              self.bounds_dict[f1_name]['upper'], 30)
        f2_range = np.linspace(self.bounds_dict[f2_name]['lower'],
                              self.bounds_dict[f2_name]['upper'], 30)

        # Mean values for other factors
        mean_values = {fname: (self.bounds_dict[fname]['lower'] +
                              self.bounds_dict[fname]['upper']) / 2
                      for fname in factor_names}

        # Predict on grid
        predictions = np.zeros((len(f2_range), len(f1_range)))
        for i, f2_val in enumerate(f2_range):
            for j, f1_val in enumerate(f1_range):
                X_test = np.array([[mean_values[fname] if idx != factor1_idx and idx != factor2_idx
                                  else (f1_val if idx == factor1_idx else f2_val)
                                  for idx, fname in enumerate(factor_names)]])
                predictions[i, j] = self.gp_model.predict(X_test)[0]

        if plot_type == "heatmap":
            fig = go.Figure(data=go.Heatmap(
                x=f1_range, y=f2_range, z=predictions,
                colorscale='Viridis',
                name='Target',
                colorbar=dict(title="Target")
            ))

            # Add observed data points
            if self.X_train is not None and len(self.X_train[0]) >= max(factor1_idx, factor2_idx) + 1:
                fig.add_trace(go.Scatter(
                    x=self.X_train[:, factor1_idx],
                    y=self.X_train[:, factor2_idx],
                    mode='markers',
                    marker=dict(color='white', size=10, symbol='x', line=dict(width=2)),
                    name='Observed'
                ))

            # ========================================================================
            # ADD SUGGESTED POINTS MARKERS TO HEATMAP
            # ========================================================================
            if self.suggested_points_df is not None and len(self.suggested_points_df) > 0:

                # Extract suggested points for these two factors
                if f1_name in self.suggested_points_df.columns and f2_name in self.suggested_points_df.columns:
                    suggested_factor1 = self.suggested_points_df[f1_name].values
                    suggested_factor2 = self.suggested_points_df[f2_name].values

                    # Add scatter plot for suggested points
                    fig.add_trace(go.Scatter(
                        x=suggested_factor1,
                        y=suggested_factor2,
                        mode='markers',
                        marker=dict(
                            size=12,
                            symbol='x',
                            color='red',
                            line=dict(width=2, color='darkred')
                        ),
                        name='BO Suggestions',
                        text=[f"Rank {i+1}<br>Score: {score:.3f}"
                              for i, score in enumerate(self.suggested_points_df['Acquisition_Value'].values)],
                        hovertemplate='<b>Suggested Point</b><br>' +
                                      f'{f1_name}: %{{x:.4f}}<br>' +
                                      f'{f2_name}: %{{y:.4f}}<br>' +
                                      '%{text}<extra></extra>',
                        showlegend=True
                    ))

            fig.update_layout(
                title=f"2D Acquisition: {f1_name} vs {f2_name}",
                xaxis_title=f1_name,
                yaxis_title=f2_name,
                template='plotly_white',
                width=800,
                height=800,  # Same as width = square plot!
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
        else:  # 3D surface
            fig = go.Figure(data=go.Surface(
                x=f1_range, y=f2_range, z=predictions,
                colorscale='Viridis',
                name='Target'
            ))

            # ========================================================================
            # ADD SUGGESTED POINTS MARKERS TO SURFACE (RED DOTS)
            # ========================================================================
            if self.suggested_points_df is not None and len(self.suggested_points_df) > 0:
                # Extract suggested points for these two factors
                if f1_name in self.suggested_points_df.columns and f2_name in self.suggested_points_df.columns:
                    suggested_factor1 = self.suggested_points_df[f1_name].values
                    suggested_factor2 = self.suggested_points_df[f2_name].values

                    # Get predicted z-values for these points
                    suggested_z = self.suggested_points_df['Expected_Target'].values

                    # Add scatter3d for suggested points
                    fig.add_trace(go.Scatter3d(
                        x=suggested_factor1,
                        y=suggested_factor2,
                        z=suggested_z,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color='red',
                            symbol='circle',
                            line=dict(width=2, color='darkred')
                        ),
                        name='BO Suggestions',
                        text=[f"Rank {i+1}<br>Target: {target:.3f}<br>Acq: {score:.3f}"
                              for i, (target, score) in enumerate(zip(
                                  self.suggested_points_df['Expected_Target'].values,
                                  self.suggested_points_df['Acquisition_Value'].values))],
                        hovertemplate='<b>Suggested Point</b><br>' +
                                      f'{f1_name}: %{{x:.4f}}<br>' +
                                      f'{f2_name}: %{{y:.4f}}<br>' +
                                      'Target: %{z:.4f}<br>' +
                                      '%{text}<extra></extra>'
                    ))

            fig.update_layout(
                title=f"3D Acquisition: {f1_name} vs {f2_name}",
                scene=dict(
                    xaxis_title=f1_name,
                    yaxis_title=f2_name,
                    zaxis_title='Target'
                ),
                template='plotly_white',
                height=600
            )

        return fig

    def export_suggested_points(self, filename: str = "bo_suggested_experiments.csv") -> str:
        """
        Export suggested experimental points to CSV file.

        Args:
            filename: Output CSV filename

        Returns:
            Path to exported file

        Raises:
            ValueError: If no suggested points available
        """
        if self.suggested_points_df is None:
            raise ValueError("No suggested points available. Run BO first using run_bayesian_optimization().")

        self.suggested_points_df.to_csv(filename, index=False)
        return filename

    def get_optimization_summary(self) -> Dict:
        """
        Return summary statistics of optimization.

        Returns:
            Dictionary with summary information
        """
        summary = {
            'n_factors': len(self.bounds_dict),
            'factor_names': list(self.bounds_dict.keys()),
            'n_constraints': len(self.constraints),
            'n_existing_experiments': len(self.data),
            'n_suggested_points': len(self.suggested_points_df) if self.suggested_points_df is not None else 0,
            'gp_model_score': self.gp_score if hasattr(self, 'gp_score') else None,
            'optimization_history': self.optimization_history,
            'X_cols': self.X_cols if hasattr(self, 'X_cols') else None,
            'y_col': self.y_col if hasattr(self, 'y_col') else None
        }

        # Add bounds info
        if self.bounds_dict:
            summary['bounds_summary'] = {
                fname: f"[{bounds['lower']:.2f}, {bounds['upper']:.2f}]"
                for fname, bounds in self.bounds_dict.items()
            }

        return summary


# Utility functions for Streamlit integration
def initialize_bo_from_session_state() -> Optional[BayesianOptimizationDesigner]:
    """
    Initialize BO designer from Streamlit session state.

    Returns:
        BayesianOptimizationDesigner instance or None if data not available
    """
    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        return None

    workspace_meta = st.session_state.get('workspace_metadata', {})
    bo_designer = BayesianOptimizationDesigner(
        st.session_state.current_data,
        workspace_metadata=workspace_meta
    )

    return bo_designer


def display_bo_summary(bo_designer: BayesianOptimizationDesigner):
    """
    Display optimization summary in Streamlit.

    Args:
        bo_designer: BayesianOptimizationDesigner instance
    """
    summary = bo_designer.get_optimization_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Factors", summary['n_factors'])
    with col2:
        st.metric("Existing Experiments", summary['n_existing_experiments'])
    with col3:
        st.metric("Suggested Points", summary['n_suggested_points'])

    if summary['gp_model_score'] is not None:
        st.info(f"GP Model R² Score: {summary['gp_model_score']:.4f}")

    if summary.get('bounds_summary'):
        with st.expander("Factor Bounds"):
            for fname, bounds_str in summary['bounds_summary'].items():
                st.write(f"**{fname}**: {bounds_str}")


if __name__ == "__main__":
    print("Bayesian Optimization Designer Module")
    print("=" * 50)
    print("Ready to import from Streamlit pages")
    print("\nExample usage:")
    print("""
    from bayesian_optimization_doe import BayesianOptimizationDesigner

    # Initialize with workspace data
    bo = BayesianOptimizationDesigner(st.session_state.current_data)

    # Validate data
    valid, msg = bo.validate_workspace_data()

    # Set factor bounds
    bo.set_factor_bounds('Temperature', 20, 100)
    bo.set_factor_bounds('Pressure', 1, 10)

    # Fit GP model
    bo.fit_gaussian_process_from_data(['Temperature', 'Pressure'], 'Yield')

    # Run optimization
    suggestions = bo.run_bayesian_optimization(batch_size=5)

    # Visualize
    fig = bo.visualize_acquisition_1d(0)
    st.plotly_chart(fig)
    """)
