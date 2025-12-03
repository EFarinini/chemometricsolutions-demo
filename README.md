# ChemometricSolutions - Modular Web Application

Professional chemometric analysis tools brought to the web. A comprehensive Streamlit-based platform for PCA, MLR/DoE, data handling, and classification with a fully modular architecture featuring root-level menu modules and shared workspace utilities.

**Live Demo:** https://chemometricsolutions-demo.streamlit.app/
**GitHub:** https://github.com/EFarinini/chemometricsolutions-demo

---

**ChemometricSolutions** - Making Professional Chemometric Analysis Accessible to Everyone 🧪📊✨

---
## 📂 Project Structure

```
chemometricsolutions-demos/
│
├── streamlit_app.py                 # Main entry point (initializes app state)
├── homepage.py                      # Homepage with navigation dashboard
├── requirements.txt                 # Python dependencies
│
├── 📊 ROOT-LEVEL MENU MODULES (Main Pages)
│   ├── data_handling.py            # Data import/export/transformation
│   ├── pca.py                      # Principal Component Analysis
│   ├── mlr_doe.py                  # Multiple Linear Regression & Design of Experiments
│   ├── multi_doe_page.py           # Advanced multi-factor DoE
│   ├── transformations.py          # Data preprocessing & spectral preprocessing
│   ├── pca_monitoring_page.py      # Quality Control & Statistical Process Monitoring
│   ├── bayesian_optimization_page.py # Bayesian Optimization for experimental design
│   ├── classification_page.py      # Classification algorithms (PLS-DA, SIMCA, LDA, KNN)
│   ├── calibration_page.py         # PLS Multivariate Calibration
│   └── univariate_page.py          # Univariate statistical analysis
│
├── 🔧 COMMON UTILITIES (Root-level shared resources)
│   ├── color_utils.py              # Color palettes & visualization theme management
│   ├── workspace_utils.py          # Workspace management & dataset activation
│   └── config.py                   # Global configuration settings
│
└── 📁 MODULES FOLDER (Calculation & Computation Engines)
    ├── __init__.py
    │
    ├── data_handling/              # Data I/O operations (backend for data_handling.py)
    │   ├── __init__.py
    │   ├── loaders.py              # Load CSV, Excel, RAW, DAT, SAM files
    │   ├── exporters.py            # Export data (Excel, CSV, pickle)
    │   ├── transformations.py      # Row/column operations, filtering, reshaping
    │   ├── validators.py           # Input validation & error handling
    │   └── conversions.py          # Format conversions & spectral data handling
    │
    ├── pca/                        # Principal Component Analysis (backend for pca.py)
    │   ├── __init__.py
    │   ├── calculations.py         # Core PCA, NIPALS, Varimax/Promax rotations
    │   ├── diagnostics.py          # T² (Hotelling), Q (SPE) statistics, contributions
    │   ├── plots.py                # 2D/3D scores, loadings, biplot, scree plots
    │   ├── statistics.py           # Variance explained, eigenvalues, cumulative variance
    │   ├── monitoring.py           # PCA monitoring, control charts
    │   ├── predictions.py          # Project new samples onto PCA model
    │   └── model_export.py         # Save/load PCA models
    │
    ├── mlr_doe/                    # MLR & DoE (backend for mlr_doe.py, multi_doe_page.py)
    │   ├── __init__.py
    │   ├── doe_generator.py        # Generate factorial designs (2^k, 3^k, mixed-level)
    │   ├── mlr_model.py            # MLR computation, coefficients, model equations
    │   ├── diagnostics.py          # VIF, residuals, R², RMSE, lack-of-fit tests
    │   ├── response_surface.py     # Response surface methodology, 3D surface visualization
    │   ├── candidate_points.py     # Optimal experimental point selection
    │   ├── confidence_intervals.py # Prediction intervals, uncertainty quantification
    │   ├── pareto_optimization.py  # Pareto front analysis for multi-objective optimization
    │   ├── surface_analysis.py     # Ridge analysis, optimal regions
    │   └── model_computation.py    # Model persistence & computation caching
    │
    ├── transformations/            # Data preprocessing (backend for transformations.py)
    │   ├── __init__.py
    │   ├── scaling.py              # Standardization (z-score), normalization, autoscaling
    │   ├── centering.py            # Mean centering, column-wise centering
    │   ├── spectral.py             # SNV, MSC, 1st/2nd derivatives, Savitzky-Golay
    │   ├── missing_data.py         # Missing value imputation & reconstruction
    │   ├── column_transforms.py    # Log, sqrt, box-cox, polynomial transforms
    │   ├── row_transforms.py       # Row normalization, outlier detection
    │   ├── transform_plots.py      # Before/after transformation visualizations
    │   └── preset_pipelines.py     # Pre-built transformation workflows
    │
    ├── quality_control/            # Statistical Process Monitoring (backend for pca_monitoring_page.py)
    │   ├── __init__.py
    │   ├── pca_monitoring.py       # PCA monitoring model training
    │   ├── control_charts.py       # T² and Q control chart generation
    │   ├── fault_detection.py      # Fault detection & diagnostics
    │   ├── contributions.py        # Contribution plots for T² and Q
    │   ├── limits.py               # Control limit calculations (95%, 99% confidence)
    │   └── performance.py          # False alarm rates, sensitivity analysis
    │
    ├── bayesian_optimization/      # BO for experimental design (backend for bayesian_optimization_page.py)
    │   ├── __init__.py
    │   ├── gaussian_process.py     # GP model computation
    │   ├── acquisition.py          # Acquisition functions (EI, UCB, POI)
    │   ├── optimization.py         # Point optimization & candidate generation
    │   ├── sampling.py             # Initial design sampling strategies
    │   └── convergence.py          # Convergence diagnostics & convergence plots
    │
    ├── classification/             # Pattern recognition (backend for classification_page.py)
    │   ├── __init__.py
    │   ├── models.py               # PLS-DA, SIMCA, LDA, KNN classifiers
    │   ├── training.py             # Model training, cross-validation, hyperparameter tuning
    │   ├── evaluation.py           # Accuracy, precision, recall, F1, confusion matrix, ROC
    │   ├── plots.py                # Classification scores, class boundaries, ROC curves
    │   ├── diagnostics.py          # Feature importance, model reliability, confusion analysis
    │   └── predictions.py          # New sample classification & probability estimates
    │
    ├── calibration/                # PLS Calibration (backend for calibration_page.py)
    │   ├── __init__.py
    │   ├── pls_regression.py       # PLS1/PLS2 model computation, X/Y loadings & scores
    │   ├── calibration.py          # Model calibration, cross-validation, LV selection
    │   ├── predictions.py          # Sample predictions, prediction intervals, UQ
    │   ├── diagnostics.py          # Model quality (R², RMSEC, RMSECV, RMSEP), outlier detection
    │   ├── leverage_analysis.py    # Leverage, Mahalanobis distance, prediction reliability
    │   └── model_export.py         # Save/load PLS models
    │
    ├── univariate/                 # Univariate statistics (backend for univariate_page.py)
    │   ├── __init__.py
    │   ├── descriptive_stats.py    # Mean, median, std, skewness, kurtosis
    │   ├── hypothesis_tests.py     # t-test, ANOVA, Mann-Whitney, Kruskal-Wallis
    │   ├── distributions.py        # Distribution fitting, normality tests
    │   ├── correlation.py          # Pearson, Spearman correlation matrices
    │   ├── plots.py                # Histograms, box plots, Q-Q plots, scatter matrices
    │   └── outlier_detection.py    # IQR, Z-score, Mahalanobis distance methods
    │
    └── visualization/              # Unified plotting system (used by all modules)
        ├── __init__.py
        ├── colors.py               # ChemometricSolutions color palette & theme management
        ├── plots_common.py         # Base Plotly functions, grid layouts, common formatting
        ├── themes.py               # Consistent plot styling, font settings, color schemes
        └── export_utils.py         # Plot export (PNG, SVG, PDF)
```

---

## 🎯 Workspace & Dataset Management

### **Shared Workspace System** (Root-level utilities)

The application uses a **common workspace** for managing datasets across all modules:

#### **workspace_utils.py**
- `get_workspace_datasets()` - Retrieve all datasets currently in workspace
- `activate_dataset_in_workspace(name, data)` - Set the active dataset
- `get_current_dataset()` - Retrieve the currently active dataset
- `remove_dataset_from_workspace(name)` - Remove a dataset from workspace
- `export_workspace_backup()` - Export all workspace datasets

**Usage Example:**
```python
from workspace_utils import get_current_dataset, activate_dataset_in_workspace
import pandas as pd

# Get current active dataset from workspace
data = get_current_dataset()

# Switch to different dataset
datasets = get_workspace_datasets()
if "my_dataset" in datasets:
    activate_dataset_in_workspace("my_dataset", datasets["my_dataset"])
```

#### **Dataset Flow:**
1. **Data Handling module** imports CSV/Excel → stored in workspace
2. **All other modules** access the same dataset via `workspace_utils.get_current_dataset()`
3. **Sidebar dataset selector** allows switching between loaded datasets
4. **Persistent across modules** - No need to re-import for each analysis

---

## 🔧 Root-Level Menu Modules

### **1. data_handling.py** - Data Import, Export & Management
**Entry Point:** `Main Menu → Data Handling`

**Features:**
- Load CSV, Excel (.xlsx, .xls), RAW (Bruker, JASCO, Perkin-Elmer), DAT, SAM files
- Export to Excel, CSV, pickle formats
- Data preview with statistics (samples, variables, memory usage)
- Row/column transformations, filtering, reshaping
- Workspace dataset management
- Data validation and error reporting

**Backend Connection:** `modules/data_handling/` (loaders.py, exporters.py, transformations.py)

---

### **2. pca.py** - Principal Component Analysis
**Entry Point:** `Main Menu → PCA`

**Features:**
- Complete PCA workflow (centering, scaling, SVD computation)
- Interactive 2D/3D score plots with hovering info
- Loading plots and biplot visualization
- Variance explained analysis with Scree plots
- Hotelling's T² and Q (SPE) statistics
- Contribution analysis for outlier diagnostics
- Varimax/Promax rotation
- Model diagnostics and summary statistics

**Backend Connection:** `modules/pca/` (calculations.py, plots.py, diagnostics.py, statistics.py)

---

### **3. mlr_doe.py** - Multiple Linear Regression & Design of Experiments
**Entry Point:** `Main Menu → MLR/DOE`

**Features:**
- Candidate point generation for experimental design
- Full factorial design generation (2^k, 3^k)
- MLR model computation with/without interactions
- Response surface 3D visualization
- Model diagnostics (R², adjusted R², RMSE, VIF, lack-of-fit)
- Prediction intervals and uncertainty quantification
- Automatic model equation generation

**Backend Connection:** `modules/mlr_doe/` (doe_generator.py, mlr_model.py, response_surface.py, diagnostics.py, candidate_points.py, confidence_intervals.py)

---

### **4. multi_doe_page.py** - Multi-Response Design of Experiments
**Entry Point:** `Main Menu → Multi-DOE`

**Features:**
- Define X variables once, multiple Y variables simultaneously
- Automatic model fitting for each response variable
- Unified coefficients comparison across all responses
- Parallel response surface analysis (one surface per Y)
- Multi-criteria decision making with Pareto front optimization
- Model diagnostics for each response independently
- Predictions across all models with confidence intervals
- Experimental design matrix generation (standalone tool)
- Export all model results and predictions

**Architecture:**
- Fits multiple **independent MLR models** (one per Y variable)
- All models use the **same X matrix and terms**
- Comparison views show model coefficients side-by-side
- Each response has its own diagnostic plots, surfaces, and predictions

**Backend Modules:**
- `modules/mlr_doe/` - Core MLR computation (reused for each Y)
- `mlr_utils/model_computation_multidoe.py` - Multi-response fitting engine
- `mlr_utils/model_diagnostics_multidoe.py` - Parallel diagnostics
- `mlr_utils/surface_analysis_multidoe.py` - Multi-surface visualization
- `mlr_utils/predictions_multidoe.py` - Unified predictions interface
- `mlr_utils/pareto_ui_multidoe.py` - Multi-objective optimization
- `mlr_utils/export_multidoe.py` - Batch export for all models

**Tabs in Multi-DOE Module:**
1. **Model Computation** - Select X/Y variables, fit all models, view coefficients
2. **Model Diagnostics** - R², RMSE, VIF, residuals (switchable by response)
3. **Surface Analysis** - 3D response surfaces (one per Y variable)
4. **Predictions** - Predict multiple responses simultaneously
5. **Multi-Criteria Decision Making** - Pareto front, desirability functions
6. **Generate Matrix** - Standalone experimental design tool
7. **Export** - Download models, predictions, and reports

---

### **5. transformations.py** - Data Preprocessing & Spectral Processing
**Entry Point:** `Main Menu → Transformations`

**Features:**
- Scaling methods (standardization, normalization, autoscaling)
- Mean centering
- Spectral preprocessing (SNV, MSC, Savitzky-Golay derivatives)
- Missing value imputation
- Transformation visualization (before/after)
- Preset transformation pipelines

**Backend Connection:** `modules/transformations/` (scaling.py, spectral.py, missing_data.py, transform_plots.py)

---

### **6. pca_monitoring_page.py** - Quality Control & Statistical Process Monitoring
**Entry Point:** `Main Menu → Quality Control`

**Features:**
- PCA monitoring model from historical data
- T² and Q control chart visualization
- Real-time sample monitoring against limits
- Automatic pretreatment method detection
- Contribution analysis for fault diagnostics
- Control limit calculation (95%, 99% confidence)

**Backend Connection:** `modules/quality_control/` (pca_monitoring.py, control_charts.py, fault_detection.py, contributions.py)


---

### **7. bayesian_optimization_page.py** - Bayesian Optimization
**Entry Point:** `Main Menu → Bayesian Optimization`

**Features:**
- Gaussian Process modeling of response surface
- Acquisition function optimization (EI, UCB, POI)
- Automated optimal point suggestion
- 1D/2D/nD visualization
- Iterative refinement with new experimental data
- Convergence diagnostics

**Backend Connection:** `modules/bayesian_optimization/` (gaussian_process.py, acquisition.py, optimization.py)

---

### **8. classification_page.py** - Classification & Pattern Recognition
**Entry Point:** `Main Menu → Classification`

**Features:**
- PLS-DA, SIMCA, LDA, KNN classifiers
- Model training with cross-validation
- Hyperparameter tuning
- Performance metrics (accuracy, precision, recall, F1, ROC)
- Confusion matrix visualization
- Feature importance analysis
- New sample classification

**Backend Connection:** `modules/classification/` (models.py, training.py, evaluation.py, plots.py)

---

### **9. calibration_page.py** - PLS Multivariate Calibration
**Entry Point:** `Main Menu → PLS Calibration`

**Features:**
- PLS1/PLS2 model computation
- Cross-validation with LV selection
- Prediction intervals and uncertainty quantification
- Model quality metrics (R², RMSEC, RMSECV, RMSEP)
- Outlier detection (Mahalanobis distance, leverage)
- Sample predictions with confidence bands

**Backend Connection:** `modules/calibration/` (pls_regression.py, calibration.py, predictions.py, diagnostics.py)

---

### **10. univariate_page.py** - Univariate Statistical Analysis
**Entry Point:** `Main Menu → Univariate Analysis`

**Features:**
- Descriptive statistics (mean, median, std, skewness, kurtosis)
- Hypothesis testing (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis)
- Distribution fitting and normality tests
- Correlation matrices (Pearson, Spearman)
- Univariate visualizations (histograms, Q-Q plots, box plots)
- Outlier detection methods (IQR, Z-score, Mahalanobis)

**Backend Connection:** `modules/univariate/` (descriptive_stats.py, hypothesis_tests.py, plots.py, outlier_detection.py)

---

## 🔨 Common Utilities (Root-Level)

### **color_utils.py** - Theme & Color Management
```python
from color_utils import get_theme_colors, apply_streamlit_theme

# Get ChemometricSolutions color palette
colors = get_theme_colors()
primary_blue = colors['primary']
accent_orange = colors['accent']

# Apply theme to Streamlit app
apply_streamlit_theme()
```

**Contains:**
- ChemometricSolutions brand colors (primary blue #2E5293, accent orange #FF6B35)
- Color palettes for plots (discrete, continuous, diverging)
- Theme management functions
- Accessibility-compliant color selections

---

### **workspace_utils.py** - Dataset Management
```python
from workspace_utils import get_current_dataset, activate_dataset_in_workspace

# Access shared dataset
data = get_current_dataset()

# Switch dataset
all_datasets = get_workspace_datasets()
if "backup_data" in all_datasets:
    activate_dataset_in_workspace("backup_data", all_datasets["backup_data"])
```

**Key Functions:**
- `get_workspace_datasets()` - Dict of all workspace datasets
- `get_current_dataset()` - Currently active dataset
- `activate_dataset_in_workspace(name, dataframe)` - Switch active dataset
- `remove_dataset_from_workspace(name)` - Remove dataset
- `get_dataset_info(name)` - Dataset metadata

---

### **config.py** - Global Configuration
```python
import config

# Access app settings
APP_NAME = config.APP_NAME
THEME_COLOR = config.PRIMARY_COLOR
MAX_UPLOAD_SIZE = config.MAX_FILE_SIZE_MB
```

---

## 🎨 Visualization Module

All plotting is centralized in `modules/visualization/`:

```python
from modules.visualization import plots_common, colors

# Create Plotly figure with ChemometricSolutions theme
theme = colors.get_chemometric_theme()
fig = plots_common.create_blank_figure(theme=theme)
fig.add_trace(...)
fig.update_layout(**theme['layout'])
```

**Unified styling ensures:** Consistent colors, fonts, sizing across all modules ✓

---

## 📊 Shared Workspace Architecture

```
Session State (Streamlit)
    ↓
workspace_utils.py (global dataset management)
    ↓
st.session_state['current_dataset'] ← Active dataset
st.session_state['all_datasets'] ← Dict of all loaded datasets
st.session_state['dataset_name'] ← Current dataset name
    ↓
Every module accesses via: get_current_dataset()
```

**Advantage:** Load data once in Data Handling → Use everywhere else. ✓

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/EFarinini/chemometricsolutions-demo.git
cd chemometricsolutions-demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Run Streamlit app
streamlit run streamlit_app.py
```

Open browser → `http://localhost:8501`

---

## 📝 Development Quick Start

### **Adding a New Feature to Existing Module**

**Example: Add confidence interval bands to PCA predictions**

1. **Backend calculation** → `modules/pca/predictions.py`
   ```python
   def predict_with_ci(pca_model, new_data, confidence=0.95):
       # Implementation here
       return scores, ci_lower, ci_upper
   ```

2. **Frontend UI** → `pca.py`
   ```python
   from modules.pca.predictions import predict_with_ci
   
   # In your Streamlit tab:
   scores, ci_lower, ci_upper = predict_with_ci(model, new_data)
   fig = plots.plot_scores_with_ci(scores, ci_lower, ci_upper)
   st.plotly_chart(fig)
   ```

### **Creating a New Module (Advanced)**

1. Create folder `modules/mymodule/`
2. Implement calculation functions (no Streamlit!)
3. Create root-level file `mymodule_page.py` with `show()` function
4. Add import check to `homepage.py`
5. Add button/link to homepage and sidebar navigation
6. Update this README

---

## 🎯 Architecture Best Practices

✅ **Separation of Concerns:**
- Calculation logic in `modules/`
- UI logic in root-level `.py` files
- Common utilities in root-level `*_utils.py`

✅ **Module Independence:**
- Each module folder can work standalone
- No circular dependencies
- Shared imports only through `modules/visualization/` and utils

✅ **Workspace Integration:**
- All data flows through workspace
- No hardcoded file paths
- Session state synchronization across modules

✅ **Code Reusability:**
- Calculation functions reusable in other projects
- Plotting functions consistent across all modules
- Utility functions generic and well-documented

---

## 📊 Features Matrix

| Feature | Module | Status |
|---------|--------|--------|
| Data Import/Export | data_handling.py | ✅ Active |
| PCA Analysis | pca.py | ✅ Active |
| PCA Monitoring | pca_monitoring_page.py | ✅ Active |
| MLR & DoE | mlr_doe.py | ✅ Active |
| Multi-Response DoE | multi_doe_page.py | ✅ Active |
| Data Preprocessing | transformations.py | ✅ Active |
| Bayesian Optimization | bayesian_optimization_page.py | ✅ Active |
| Classification | classification_page.py | ✅ Active |
| PLS Calibration | calibration_page.py | ✅ Active |
| Univariate Stats | univariate_page.py | ✅ Active |

---

## 💻 Technology Stack

- **Framework:** Streamlit 1.28+
- **Scientific Computing:** NumPy, SciPy, scikit-learn
- **Data Manipulation:** Pandas
- **Visualization:** Plotly, Matplotlib
- **Deployment:** Streamlit Cloud
- **Python:** 3.9+ (tested on 3.13)

---

## 📚 Documentation

- **Module-Specific Docs:** README.md in each `modules/*/`
- **API Reference:** Docstrings in each function
- **Examples:** See `examples/` folder (if present)
- **Theory:** See `docs/theory.md` (if present)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow code style (PEP 8, type hints, docstrings)
4. Separate calculation logic from UI
5. Update this README
6. Submit Pull Request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👨‍🔬 Author

**Dr. Emanuele Farinini, PhD**  
Chemometrics & Analytical Chemistry Expert

- Website: https://chemometricsolutions.com
- GitHub: https://github.com/FarininiChemometricSolutions
- Email: chemometricsolutions@gmail.com

---

## 🙏 Acknowledgments

Built with ❤️ using:
- Python, Streamlit, Plotly
- scikit-learn, SciPy, NumPy, Pandas
- Reference implementations: R packages, CAT software, chemometrics literature

---

## 📞 Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions  
- **Email:** chemometricsolutions@gmail.com
- **Live Demo:** https://chemometricsolutions-demo.streamlit.app