# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ChemometricSolutions Interactive Demos** - A Streamlit-based web application for chemometric analysis and multivariate statistics, specifically designed for spectroscopy and analytical chemistry applications.

**Key Domains**: NIR spectroscopy, PCA analysis, Design of Experiments (DoE), Multiple Linear Regression (MLR), data preprocessing/transformations

## Running the Application

```bash
# Run the main application
streamlit run streamlit_app.py

# The app will be available at http://localhost:8501
```

## Development Environment

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

### Entry Point & Navigation
- **streamlit_app.py**: Minimal entry point, imports and runs `homepage.main()`
- **homepage.py**: Main application controller with page routing, sidebar navigation, and home page display. Uses session state (`st.session_state.current_page`) for page management.

### Core Module Structure

The application follows a **page-based architecture** where each major feature is a separate module:

1. **data_handling.py** (~1600 lines)
   - Multi-format file import/export (CSV, Excel, SAM, RAW, DAT, spectroscopy formats)
   - Workspace management for datasets and transformations
   - Dataset operations (transpose, randomize, metadata classification)
   - **Key Session State**: `current_data`, `current_dataset`, `transformation_history`, `split_datasets`

2. **pca.py** (~4400 lines)
   - Complete PCA workflow: computation, variance analysis, scores/loadings visualization
   - Standard PCA and Varimax rotation
   - Advanced diagnostics: T², Q statistics, control charts
   - Dataset splitting and workspace integration
   - **Key Session State**: `pca_model`, `pca_results`, `split_datasets`

3. **mlr_doe.py** (~1800 lines)
   - Multiple Linear Regression with interactions and quadratic terms
   - Design of Experiments: candidate points, factorial designs
   - Statistical diagnostics: VIF, leverage, lack of fit tests, experimental replicates
   - Central point detection and validation
   - **Key Session State**: `mlr_model`, `mlr_y_var`, `mlr_x_vars`, `mlr_central_points`

4. **transformations.py** (~1000 lines)
   - Row transformations: SNV, derivatives, Savitzky-Golay, moving average, binning
   - Column transformations: autoscaling, DoE coding, log, range scaling
   - Visual comparison plots with categorical/quantitative color mapping
   - Auto-saves to `transformation_history`

### Supporting Modules

- **color_utils.py**: Unified color scheme system for consistent theming across all visualizations. Handles both categorical (discrete colors) and quantitative (blue-to-red gradient) variables.

- **pca_ai_agent.py**: AI-powered loadings interpretation using Claude API (via datapizza-ai)
- **pca_ai_utils.py**: OpenAI integration for PCA loadings analysis

## Session State Architecture

Critical session state variables used across modules:

```python
st.session_state.current_data          # Active DataFrame
st.session_state.current_dataset       # Name of active dataset
st.session_state.transformation_history  # Dict: {name: {data, transform, params, timestamp}}
st.session_state.split_datasets        # Dict: {name: {data, type, parent, n_samples}}
st.session_state.custom_variables      # Dict: {name: Series} - user-created variables
st.session_state.pca_model            # PCA model object and results
st.session_state.mlr_model            # MLR regression results
```

**Data Flow Pattern**:
1. Data loaded in `data_handling.py` → stored in `current_data`
2. Transformations create new entries in `transformation_history`
3. PCA creates `split_datasets` for selected samples
4. All modules read from `current_data` and can update workspace state

## Key Design Patterns

### Transformation Workflow
When applying transformations:
1. Always preserve original dataset structure (metadata columns)
2. Only transform numeric columns in specified range
3. Store transformation metadata: `{data, transform, params, col_range, timestamp, original_dataset}`
4. Auto-save originals with `_ORIGINAL` suffix

### Dataset Naming Convention
- Original: `filename.csv` or `Dataset_Name`
- Transformed: `base_name.transform_code` (e.g., `NIR_Data.snv`, `DoE_Matrix.cod`)
- Splits: `descriptive_name` (e.g., `Training_Set`, `Selected_Samples_PC1vsPC2`)

### Color Mapping System
All visualizations use `color_utils.py`:
- **Categorical variables**: Discrete colors from predefined palette
- **Quantitative variables**: Continuous blue-to-red gradient
- Check variable type with `is_quantitative_variable(data)`
- Use `create_categorical_color_map(unique_values)` or `create_quantitative_color_map(values)`

## File Format Support

**Spectroscopy formats** (in `data_handling.py`):
- **SAM files**: NIR spectra, MNIR format detection
- **RAW files**: XRD diffraction data, binary parsing
- **DAT/ASC files**: General spectral data with transpose option
- **Standard formats**: CSV, Excel, JSON, TXT

**Format-specific handling**:
- Spectral data often requires transpose (variables×samples → samples×variables)
- Wavelength detection for column naming
- Metadata extraction from file headers

## Statistical Algorithms

### PCA Implementation
- Uses sklearn's `PCA` with manual SVD option
- Varimax rotation via custom implementation
- T² statistic: Mahalanobis distance in PC space with F-distribution limits
- Q statistic: Squared prediction error with Jackson-Mudholkar limits
- Cross-validation: Leave-one-out with RMSECV and Q²

### MLR/DoE Implementation
- Manual matrix algebra: `b = (X'X)^-1 X'y`
- VIF calculation matches R formula: `sum(X_centered_i^2) * diag(XtX_inv)_i`
- Lack of fit test: Compares model error vs pure experimental error from replicates
- Central point detection: All variables at midpoint (0 for coded variables)
- Experimental replicate detection: Identifies duplicate experimental conditions

## Data Preprocessing Notes

### Variable Selection Best Practices
1. **Spectral data**: Auto-detect numeric columns as wavelengths (numeric column names in range 200-25000)
2. **Metadata**: Non-numeric column names or values outside spectroscopic ranges
3. Use `Metadata Management` tab in Data Handling for classification

### Transformation Guidelines
- **SNV**: Use for scatter correction in NIR spectroscopy
- **Derivatives**: Remove baseline effects, but reduce data points
- **Savitzky-Golay**: Smoothing + derivatives, preserves peak shapes
- **DoE Coding [-1,1]**: Required for proper MLR interaction terms
- **Autoscaling**: Standard for PCA when variables have different units

## Common Development Tasks

### Adding a New Transformation
1. Add transformation function in `transformations.py` (row or column section)
2. Add to transforms dictionary in `show()` function
3. Include parameter inputs if needed
4. Ensure proper handling of shape changes (derivatives reduce dimensions)
5. Test with both spectral and non-spectral data

### Adding a New Statistical Method
1. Create module following pattern: `method_name.py` with `show()` function
2. Add import in `homepage.py` with availability flag
3. Add navigation button in sidebar
4. Ensure session state compatibility with existing workspace
5. Document key session state variables

### Modifying Visualizations
All plots use Plotly with consistent theming via `color_utils.py`:
- Get scheme: `schemes = get_unified_color_schemes()`
- Apply to layout: `fig.update_layout(plot_bgcolor=schemes['background'])`
- Color data points: Use `create_categorical_color_map()` or `create_quantitative_color_map()`

## Testing & Validation

When modifying statistical functions:
1. Test with sample NIR data (included in homepage)
2. Verify against R/MATLAB reference implementations
3. Check edge cases: single sample, all missing values, rank-deficient matrices
4. Validate numerical accuracy for key statistics (R², VIF, T²/Q limits)

## External Dependencies

**Core**: streamlit, pandas, numpy, scipy, scikit-learn, plotly
**File handling**: openpyxl (Excel), xlrd (legacy Excel)
**Optional AI features**: datapizza-ai, datapizza-ai-clients-openai, datapizza-ai-clients-anthropic

## Performance Considerations

- Large spectral datasets (>1000 variables): Limit plot density, use binning for visualization
- Session state size: Clear `transformation_history` periodically for long sessions
- DataFrame operations: Always use `.copy()` when modifying to avoid aliasing issues
- Plotly rendering: Limit traces to <500 for responsive interaction

## Important Conventions

1. **Index handling**: Use 1-based indexing for user-facing inputs, convert to 0-based internally
2. **Column ranges**: Inclusive notation for user (1-10), exclusive end for Python (0:10)
3. **Error messages**: Provide actionable guidance, not just error text
4. **Missing data**: Handle gracefully, inform user of automatic handling
5. **Workspace persistence**: Transformation history and splits are session-only (not saved to disk by default)

## Module Interaction Map

```
streamlit_app.py
    └── homepage.main()
        ├── data_handling.show()
        │   └── Updates: current_data, transformation_history
        ├── pca.show()
        │   ├── Reads: current_data
        │   └── Updates: pca_model, split_datasets
        ├── mlr_doe.show()
        │   ├── Reads: current_data
        │   └── Updates: mlr_model
        └── transformations.show()
            ├── Reads: current_data, transformation_history
            └── Updates: transformation_history, current_data
```

All modules use `color_utils` for consistent visualization theming.

## Notes for AI Development

- This is a **professional chemometric application** for analytical chemistry
- Statistical correctness is paramount - verify against published methods
- User base: analytical chemists, quality control labs, research scientists
- Data typical sizes: 50-500 samples, 100-2000 variables (NIR spectra)
- Professional tone: informative but concise, no unnecessary emojis unless user requests
