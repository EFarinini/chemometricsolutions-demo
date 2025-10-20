# pca_utils Package

Comprehensive Python package for Principal Component Analysis (PCA) with advanced diagnostics and visualizations.

## Overview

`pca_utils` is a modular PCA library designed for chemometric analysis and multivariate statistics. It provides:

- **Core PCA computations** with sklearn integration
- **Varimax rotation** for interpretable factor structures
- **Rich visualizations** using Plotly
- **Statistical diagnostics** (T², Q residuals, leverage, cross-validation)
- **Workspace management** for dataset splitting and persistence

## Installation

The package is part of the ChemometricSolutions project. No separate installation required - just import:

```python
from pca_utils import compute_pca, plot_scores, calculate_hotelling_t2
```

## Package Structure

```
pca_utils/
├── __init__.py              # Package exports and metadata
├── config.py                # Configuration constants
├── pca_calculations.py      # Core PCA and Varimax functions
├── pca_plots.py             # Plotly visualization functions
├── pca_statistics.py        # Statistical diagnostics
├── pca_workspace.py         # Workspace management
└── README.md               # This file
```

## Quick Start

### Basic PCA Analysis

```python
import pandas as pd
import numpy as np
from pca_utils import compute_pca, plot_scree, plot_scores

# Load your data
data = pd.DataFrame(np.random.randn(100, 20))

# Perform PCA
results = compute_pca(data, n_components=5, center=True, scale=True)

# Create visualizations
scree_fig = plot_scree(results['explained_variance_ratio'])
scores_fig = plot_scores(results['scores'], 'PC1', 'PC2',
                          results['explained_variance_ratio'])

# Access results
print(f"Variance explained: {results['cumulative_variance'][4]*100:.1f}%")
print(f"Scores shape: {results['scores'].shape}")
print(f"Loadings shape: {results['loadings'].shape}")
```

### Varimax Rotation

```python
from pca_utils import compute_pca, varimax_rotation

# Perform PCA
pca_results = compute_pca(data, n_components=5, center=True, scale=True)

# Apply Varimax rotation for interpretability
rotated_loadings, n_iter = varimax_rotation(
    pca_results['loadings'],
    max_iter=100,
    tol=1e-6
)

print(f"Varimax converged in {n_iter} iterations")
```

### Statistical Diagnostics

```python
from pca_utils import (compute_pca, calculate_hotelling_t2,
                       calculate_q_residuals, calculate_contributions)

# Perform PCA
results = compute_pca(data, n_components=5, center=True, scale=True)

# Hotelling's T² statistic (outlier detection in model space)
t2_values, t2_limit = calculate_hotelling_t2(
    results['scores'],
    results['eigenvalues'],
    alpha=0.95
)
outliers_t2 = np.where(t2_values > t2_limit)[0]
print(f"T² outliers: {len(outliers_t2)} samples")

# Q residuals (outlier detection outside model space)
q_values, q_limit = calculate_q_residuals(
    results['processed_data'],
    results['scores'],
    results['loadings'],
    alpha=0.95
)
outliers_q = np.where(q_values > q_limit)[0]
print(f"Q residuals outliers: {len(outliers_q)} samples")

# Variable contributions
contrib_df = calculate_contributions(
    results['loadings'],
    results['explained_variance_ratio'],
    n_components=5
)
print("Top 5 contributing variables:")
print(contrib_df.nlargest(5, 'Contribution_%')[['Variable', 'Contribution_%']])
```

### Cross-Validation

```python
from pca_utils import cross_validate_pca

# Determine optimal number of components
cv_results = cross_validate_pca(
    data,
    max_components=10,
    n_folds=7,
    center=True,
    scale=True
)

print(f"Optimal components: {cv_results['optimal_components']}")
print(f"Q² values: {cv_results['Q2']}")
print(f"RMSECV values: {cv_results['RMSECV']}")
```

## Module Reference

### 1. pca_calculations.py

**Core PCA computation and rotation functions.**

#### `compute_pca(X, n_components, center=True, scale=False)`

Perform Principal Component Analysis on input data.

**Parameters:**
- `X` (DataFrame or ndarray): Input data (n_samples × n_features)
- `n_components` (int): Number of components to compute
- `center` (bool): Whether to center data (default: True)
- `scale` (bool): Whether to scale to unit variance (default: False)

**Returns:**
- Dictionary with keys: `model`, `scores`, `loadings`, `explained_variance`,
  `explained_variance_ratio`, `cumulative_variance`, `eigenvalues`, `scaler`, `processed_data`

**Raises:**
- `ValueError`: If input is invalid (wrong shape, NaN values, etc.)
- `TypeError`: If n_components is not an integer

#### `varimax_rotation(loadings, max_iter=100, tol=1e-6)`

Apply Varimax rotation to PCA loadings.

**Parameters:**
- `loadings` (DataFrame or ndarray): Loading matrix (n_features × n_components)
- `max_iter` (int): Maximum iterations (default: 100)
- `tol` (float): Convergence tolerance (default: 1e-6)

**Returns:**
- Tuple: `(rotated_loadings, n_iterations)`

**Raises:**
- `ValueError`: If loadings has wrong shape or invalid parameters

### 2. pca_plots.py

**Plotly-based visualization functions.**

#### `plot_scree(explained_variance_ratio, is_varimax=False, component_labels=None)`

Create scree plot showing variance explained by each component.

#### `plot_cumulative_variance(cumulative_variance, is_varimax=False, component_labels=None, reference_lines=None)`

Create cumulative variance plot with optional reference lines.

#### `plot_scores(scores, pc_x, pc_y, explained_variance_ratio, color_by=None, text_labels=None, is_varimax=False, show_labels=False, show_convex_hull=False, hull_opacity=0.7)`

Create scores scatter plot with smart color mapping.

**Features:**
- Automatic detection of quantitative vs categorical coloring
- Blue-to-red gradient for quantitative variables
- Discrete colors for categorical variables
- Optional convex hulls for groups
- Optional text labels

#### `plot_loadings(loadings, pc_x, pc_y, explained_variance_ratio, is_varimax=False, color_by_magnitude=False)`

Create loadings scatter plot.

#### `plot_loadings_line(loadings, selected_components, is_varimax=False)`

Create line plot of loadings across variables.

#### `plot_biplot(scores, loadings, pc_x, pc_y, explained_variance_ratio, color_by=None, loading_scale=1.0, max_loadings=20, is_varimax=False)`

Create biplot combining scores and loadings.

### 3. pca_statistics.py

**Statistical diagnostics and validation metrics.**

#### `calculate_hotelling_t2(scores, eigenvalues, alpha=0.95)`

Calculate Hotelling's T² statistic for outlier detection in model space.

**Formula:** T²ᵢ = Σ(tᵢⱼ² / λⱼ)

**Returns:** `(t2_values, t2_limit)` where limit is based on F-distribution.

#### `calculate_q_residuals(X, scores, loadings, alpha=0.95)`

Calculate Q residuals (SPE) for outlier detection outside model space.

**Formula:** Qᵢ = Σ(xᵢⱼ - x̂ᵢⱼ)²

**Returns:** `(q_values, q_limit)` using Jackson-Mudholkar approximation.

#### `calculate_contributions(loadings, explained_variance_ratio, n_components=None, normalize=True)`

Calculate variable contributions to total variance explained.

**Returns:** DataFrame with `Variable`, `Contribution_%`, `Cumulative_%`, and individual PC contributions.

#### `calculate_leverage(scores)`

Calculate leverage (hat matrix diagonal) for each sample.

**Formula:** hᵢᵢ = tᵢᵀ(TᵀT)⁻¹tᵢ

**Returns:** Array of leverage values (0 to 1).

#### `cross_validate_pca(X, max_components, n_folds=7, center=True, scale=False)`

Perform k-fold cross-validation to determine optimal number of components.

**Metrics:**
- Q² (predictive ability): 1 - PRESS/TSS
- RMSECV: √(PRESS / (n × p))
- PRESS: Predicted Residual Error Sum of Squares

**Returns:** Dictionary with `n_components`, `Q2`, `RMSECV`, `PRESS`, `optimal_components`.

### 4. pca_workspace.py

**Workspace and dataset management functions.**

#### `save_workspace_to_file(filepath=None)`

Save split datasets to JSON file.

#### `load_workspace_from_file(filepath=None)`

Load split datasets from JSON file.

#### `save_dataset_split(selected_data, remaining_data, pc_x, pc_y, parent_name=None, selection_method='Manual')`

Save dataset split to Streamlit session state.

**Returns:** `(selected_name, remaining_name)`

#### `get_split_datasets_info()`

Get summary information about all split datasets in workspace.

**Returns:** Dictionary with count, dataset names, total samples, counts by type and parent.

### 5. config.py

**Package configuration constants.**

```python
DEFAULT_N_COMPONENTS = None          # Auto-determine components
DEFAULT_CONFIDENCE_LEVEL = 0.95      # 95% confidence
VARIMAX_MAX_ITER = 20                # Varimax iterations
VARIMAX_TOLERANCE = 1e-6             # Varimax convergence tolerance
```

## Error Handling

All functions include comprehensive input validation:

```python
from pca_utils import compute_pca

# Example: Invalid input handling
try:
    results = compute_pca(data, n_components=100)  # Too many components
except ValueError as e:
    print(f"Error: {e}")
    # Error: n_components (100) cannot exceed min(n_samples, n_features) = 50
```

Common errors:
- `ValueError`: Invalid shapes, NaN values, parameter ranges
- `TypeError`: Wrong parameter types (e.g., non-integer n_components)

## Testing

Run comprehensive tests:

```bash
python test_pca_refactoring.py
```

Test coverage:
- ✅ Import tests (all modules)
- ✅ Functionality tests (compute_pca, varimax_rotation)
- ✅ Plotting tests (all visualization functions)
- ✅ Statistical tests (T², Q, contributions, leverage)
- ✅ Workspace tests (session state management)
- ✅ Integration test (full workflow)

## Integration with pca.py

The `pca.py` Streamlit application uses all pca_utils modules:

```python
# In pca.py
from pca_utils.pca_calculations import compute_pca, varimax_rotation
from pca_utils.pca_plots import (plot_scores, plot_loadings, plot_biplot,
                                  plot_scree, plot_cumulative_variance,
                                  plot_loadings_line, add_convex_hulls)
from pca_utils.pca_statistics import (calculate_hotelling_t2, calculate_q_residuals,
                                       calculate_contributions, calculate_leverage,
                                       cross_validate_pca)
from pca_utils.pca_workspace import (save_workspace_to_file, load_workspace_from_file,
                                      save_dataset_split, get_split_datasets_info,
                                      delete_split_dataset, clear_all_split_datasets)

# Then use in Streamlit UI:
if st.button("Compute PCA"):
    pca_results = compute_pca(selected_data, n_components=n_pc,
                               center=True, scale=scale_data)
    st.session_state.pca_results = pca_results
```

## Best Practices

### 1. Data Preprocessing

```python
# For data with different units → use scaling
results = compute_pca(data, n_components=5, center=True, scale=True)

# For data with same units → centering only
results = compute_pca(data, n_components=5, center=True, scale=False)
```

### 2. Choosing Number of Components

Use multiple criteria:

```python
# 1. Scree plot (elbow method)
fig = plot_scree(results['explained_variance_ratio'])

# 2. Cumulative variance (e.g., 80% or 95%)
n_80 = np.where(results['cumulative_variance'] >= 0.80)[0][0] + 1

# 3. Kaiser criterion (eigenvalues > 1)
n_kaiser = np.sum(results['eigenvalues'] > 1)

# 4. Cross-validation (predictive ability)
cv_results = cross_validate_pca(data, max_components=10)
n_optimal = cv_results['optimal_components']
```

### 3. Outlier Detection

Combine T² and Q statistics:

```python
t2_values, t2_limit = calculate_hotelling_t2(scores, eigenvalues)
q_values, q_limit = calculate_q_residuals(X, scores, loadings)

# Outliers in model space (unusual within PCs)
outliers_t2 = t2_values > t2_limit

# Outliers outside model space (not well described by PCs)
outliers_q = q_values > q_limit

# Combined outliers
outliers_combined = outliers_t2 | outliers_q
```

### 4. Interpreting Varimax Results

```python
# Apply Varimax for simpler interpretation
rotated_loadings, n_iter = varimax_rotation(pca_results['loadings'])

# Sort variables by magnitude of rotated loadings
for i in range(n_components):
    pc_name = f'RC{i+1}'  # Rotated Component
    top_vars = rotated_loadings.nlargest(5, pc_name)
    print(f"\n{pc_name} - Top 5 variables:")
    print(top_vars[[pc_name]])
```

## Performance Considerations

- **Large datasets (>10,000 samples)**: Consider using `sklearn.decomposition.IncrementalPCA` for out-of-core computation
- **Many features (>1,000)**: Varimax rotation may be slow; consider reducing components first
- **Plotting**: Limit data points to <5,000 for responsive Plotly plots

## Dependencies

Required packages:
- `numpy` - Numerical computations
- `pandas` - DataFrame handling
- `scikit-learn` - PCA implementation
- `scipy` - Statistical distributions
- `plotly` - Interactive visualizations
- `streamlit` - Web application framework (for workspace functions)

## Version History

- **v1.0.0** (2025-10-13)
  - Initial release
  - Refactored from monolithic pca.py
  - Comprehensive test coverage
  - Full documentation

## License

Part of the ChemometricSolutions project.

## Support

For issues or questions:
1. Check test file: `test_pca_refactoring.py`
2. Review examples in this README
3. Consult function docstrings: `help(compute_pca)`
4. Check main application: `pca.py`

## Contributing

When adding new functions:
1. Add to appropriate module (calculations, plots, statistics, workspace)
2. Include NumPy-style docstrings with Parameters, Returns, Examples
3. Add input validation and error handling
4. Export in `__init__.py` `__all__` list
5. Add tests in `test_pca_refactoring.py`
6. Update this README

---

**Happy analyzing! 📊🔬**
