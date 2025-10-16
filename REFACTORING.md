# ChemometricSolutions - Code Refactoring Guide

## Overview

This document describes the professional modular architecture implemented to separate business logic from UI code, making the codebase more maintainable, testable, and scalable.

## New Architecture

### Directory Structure

```
chemometricsolutions-demo/
├── utils/                           # Data handling utilities
│   ├── __init__.py
│   ├── data_loaders.py             # File loading functions (CSV, Excel, SAM, RAW, etc.)
│   ├── data_exporters.py           # Export functions (SAM export, etc.)
│   └── data_workspace.py           # Workspace management (transformation history)
│
├── transforms/                      # Transformation utilities
│   ├── __init__.py
│   ├── row_transforms.py           # SNV, derivatives, Savitzky-Golay, binning
│   ├── column_transforms.py        # Centering, scaling, autoscaling, DoE coding
│   └── transform_plots.py          # Visualization functions
│
├── pca_utils/                       # PCA analysis utilities (TO BE COMPLETED)
│   ├── __init__.py
│   ├── pca_calculations.py         # PCA computation, varimax rotation
│   ├── pca_plots.py               # Scores, loadings, biplots
│   └── pca_statistics.py          # T², Q statistics, validation
│
├── mlr_utils/                       # MLR/DoE utilities (TO BE COMPLETED)
│   ├── __init__.py
│   ├── mlr_calculations.py         # Model fitting, prediction
│   ├── mlr_diagnostics.py          # VIF, lack of fit, leverage
│   └── doe_utils.py                # Candidate points, factorial designs
│
└── [Main UI modules]
    ├── data_handling.py            # UI only - imports from utils/
    ├── transformations.py          # UI only - imports from transforms/
    ├── pca.py                      # UI only - imports from pca_utils/
    └── mlr_doe.py                  # UI only - imports from mlr_utils/
```

## Completed Modules

### 1. utils/ - Data Handling Utilities

**utils/data_loaders.py** (774 lines)
- `load_csv_txt()` - CSV/TXT files with encoding detection
- `load_spectral_data()` - DAT/ASC spectral files
- `load_sam_data()` - NIR spectra (MNIR format)
- `load_raw_data()` - XRD diffraction data
- `load_excel_data()` - Excel files with parameters
- `parse_clipboard_data()` - Clipboard data parsing
- `safe_join()`, `safe_format_objects()` - Helper functions

**utils/data_exporters.py**
- `create_sam_export()` - SAM format export

**utils/data_workspace.py**
- `save_original_to_history()` - Workspace management

### 2. transforms/ - Transformation Utilities

**transforms/row_transforms.py** (183 lines)
- `snv_transform()` - Standard Normal Variate
- `first_derivative_row()` - First derivative
- `second_derivative_row()` - Second derivative
- `savitzky_golay_transform()` - Savitzky-Golay filter
- `moving_average_row()` - Moving average
- `row_sum100()` - Row normalization
- `binning_transform()` - Variable binning

**transforms/column_transforms.py** (132 lines)
- `column_centering()` - Mean centering
- `column_scaling()` - Unit variance scaling
- `column_autoscale()` - Autoscaling (center + scale)
- `column_range_01()` - [0,1] scaling
- `column_range_11()` - [-1,1] DoE coding
- `column_max100()`, `column_sum100()`, `column_length1()` - Normalizations
- `column_log()` - Log transformation
- `column_first_derivative()`, `column_second_derivative()` - Derivatives
- `moving_average_column()` - Column moving average
- `block_scaling()` - Block scaling for multi-block analysis

**transforms/transform_plots.py** (217 lines)
- `plot_comparison()` - Side-by-side visualization with categorical/quantitative coloring

## Benefits of New Architecture

### 1. Separation of Concerns
- **UI Code**: Streamlit interface, user inputs, display
- **Business Logic**: Pure functions, calculations, transformations
- **Clear Boundaries**: Easy to understand what code does what

### 2. Testability
```python
# Example: Test SNV transformation without Streamlit
import pandas as pd
from transforms.row_transforms import snv_transform

data = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
result = snv_transform(data, col_range=(0, 3))
assert result.shape == (2, 3)
```

### 3. Reusability
```python
# Use functions in scripts, notebooks, or other applications
from utils.data_loaders import load_csv_txt
from transforms.row_transforms import snv_transform

# Load data
data = load_csv_txt(file, sep=',', decimal='.', ...)

# Transform
transformed = snv_transform(data, (0, 100))
```

### 4. Maintainability
- Small, focused modules (~200 lines each)
- Clear function responsibilities
- Professional docstrings with parameters and returns
- Easy to locate and fix bugs

### 5. Scalability
- Add new transformations by creating functions in appropriate module
- Add new file formats by extending data_loaders.py
- Create new analysis modules following same pattern

## Migration Guide

### How to Update Main Modules

**Before (monolithic):**
```python
# In data_handling.py
def _load_csv_txt(uploaded_file, separator, ...):
    # 100 lines of code
    ...

# Usage
data = _load_csv_txt(file, ',', '.', ...)
```

**After (modular):**
```python
# In data_handling.py
from utils.data_loaders import load_csv_txt

# Usage
data = load_csv_txt(file, ',', '.', ...)
```

### Example: Updating data_handling.py

1. Add imports at top of file:
```python
from utils.data_loaders import (
    load_csv_txt,
    load_spectral_data,
    load_sam_data,
    load_raw_data,
    load_excel_data,
    parse_clipboard_data
)
from utils.data_exporters import create_sam_export
from utils.data_workspace import save_original_to_history
```

2. Replace function calls:
```python
# Old: data = _load_csv_txt(...)
# New: data = load_csv_txt(...)

# Old: content = _create_sam_export(...)
# New: content = create_sam_export(...)

# Old: _save_original_to_history(...)
# New: save_original_to_history(...)
```

3. Remove old function definitions (they're now in utils/)

### Example: Updating transformations.py

1. Add imports:
```python
from transforms.row_transforms import (
    snv_transform,
    first_derivative_row,
    second_derivative_row,
    savitzky_golay_transform,
    moving_average_row,
    row_sum100,
    binning_transform
)
from transforms.column_transforms import (
    column_centering,
    column_scaling,
    column_autoscale,
    # ... etc
)
from transforms.transform_plots import plot_comparison
```

2. Remove old function definitions (now in transforms/ modules)
3. Update function calls (remove prefixes if any)

## Next Steps for Complete Refactoring

### 1. Create PCA Utilities (pca_utils/)

**pca_calculations.py** should contain:
- `compute_pca()` - PCA computation
- `varimax_rotation()` - Varimax rotation
- `calculate_explained_variance()` - Variance analysis
- Extract from pca.py lines ~178-237

**pca_plots.py** should contain:
- `plot_scores()` - Scores plots
- `plot_loadings()` - Loadings plots
- `plot_biplot()` - Biplots
- `add_convex_hulls()` - Convex hull overlay
- Extract from pca.py lines ~238-324

**pca_statistics.py** should contain:
- `calculate_t2_statistic()` - Hotelling's T²
- `calculate_q_statistic()` - Q residuals
- `cross_validate_pca()` - Cross-validation
- Extract statistical functions from pca.py

### 2. Create MLR/DoE Utilities (mlr_utils/)

**mlr_calculations.py** should contain:
- `fit_mlr_model()` - Model fitting (already exists in mlr_doe.py:115)
- `predict_new_points()` - Prediction (already exists in mlr_doe.py:260)
- `create_model_matrix()` - Design matrix (already exists in mlr_doe.py:58)

**mlr_diagnostics.py** should contain:
- `calculate_vif()` - Variance Inflation Factor
- `calculate_lack_of_fit()` - Lack of fit test (already exists in mlr_doe.py:383)
- `detect_replicates()` - Replicate detection (already exists in mlr_doe.py:307)
- `calculate_leverage()` - Leverage statistics

**doe_utils.py** should contain:
- `generate_candidate_points()` - Candidate generation (already exists in mlr_doe.py:28)
- `detect_central_points()` - Central point detection (already exists in mlr_doe.py:438)
- `detect_coded_matrix()` - Coded matrix detection (already exists in mlr_doe.py:493)

### 3. Update Main Modules

For each main module (data_handling.py, transformations.py, pca.py, mlr_doe.py):
1. Add imports from new utility modules
2. Replace function calls to use imported functions
3. Remove old function definitions
4. Keep only Streamlit UI code (st.markdown, st.button, st.plotly_chart, etc.)

### 4. Create Unit Tests

Create `tests/` directory with test files:
```
tests/
├── test_data_loaders.py
├── test_transforms.py
├── test_pca_calculations.py
└── test_mlr_calculations.py
```

Example test:
```python
# tests/test_transforms.py
import pandas as pd
import numpy as np
from transforms.row_transforms import snv_transform

def test_snv_transform():
    # Create test data
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })

    # Transform
    result = snv_transform(data, col_range=(0, 3))

    # Assertions
    assert result.shape == (3, 3)
    assert np.allclose(result.mean(axis=1), 0, atol=1e-10)
    assert np.allclose(result.std(axis=1, ddof=1), 1, atol=1e-10)
```

## Code Quality Improvements

### Professional Docstrings

All utility functions now have professional docstrings:
```python
def load_csv_txt(uploaded_file, separator, decimal, encoding, ...):
    """
    Load CSV/TXT files with robust encoding detection

    Parameters:
    -----------
    uploaded_file : file-like object
        Uploaded file from Streamlit
    separator : str
        Column separator (comma, tab, etc.)
    ...

    Returns:
    --------
    pd.DataFrame : Loaded data

    Raises:
    -------
    ValueError : If file cannot be decoded
    """
```

### Type Hints (Optional Enhancement)

Consider adding type hints for better IDE support:
```python
from typing import Tuple
import pandas as pd

def snv_transform(data: pd.DataFrame, col_range: Tuple[int, int]) -> pd.DataFrame:
    """Standard Normal Variate transformation"""
    ...
```

## Performance Considerations

### Import Optimization

The new structure allows for optimized imports:
```python
# Import only what you need
from transforms.row_transforms import snv_transform

# Instead of importing entire modules
import transformations  # (old approach)
```

### Lazy Loading

Consider lazy loading for expensive imports:
```python
# In pca_utils/__init__.py
def _lazy_import():
    from .pca_calculations import compute_pca
    return compute_pca

compute_pca = None  # Will be loaded on first use
```

## Backwards Compatibility

To maintain backwards compatibility during migration:

1. **Keep old functions temporarily** with deprecation warnings:
```python
# In data_handling.py
from utils.data_loaders import load_csv_txt as _load_csv_txt_new

def _load_csv_txt(*args, **kwargs):
    """Deprecated: Use utils.data_loaders.load_csv_txt instead"""
    import warnings
    warnings.warn("_load_csv_txt is deprecated", DeprecationWarning)
    return _load_csv_txt_new(*args, **kwargs)
```

2. **Gradual migration**: Update one module at a time
3. **Testing**: Test each module after migration

## Summary

### Completed
- ✅ utils/ package (data loaders, exporters, workspace)
- ✅ transforms/ package (row/column transforms, plots)
- ✅ Professional docstrings and function signatures
- ✅ Separation of UI and business logic
- ✅ Modular, testable architecture

### To Complete
- ⏳ pca_utils/ package (extract from pca.py)
- ⏳ mlr_utils/ package (extract from mlr_doe.py)
- ⏳ Update main modules to use new imports
- ⏳ Create unit tests
- ⏳ Add type hints (optional)

### Result
A professional, maintainable codebase that:
- Separates concerns (UI vs logic)
- Is easy to test
- Is reusable across projects
- Follows industry best practices
- Scales with project growth

## Contact

For questions or assistance with completing the refactoring:
- chemometricsolutions.com
- Review CLAUDE.md for project-specific guidance
