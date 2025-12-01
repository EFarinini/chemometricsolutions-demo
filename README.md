# ChemometricSolutions - Modular Web Application

> ⚠️ **OFFICIAL REPOSITORY** ⚠️  
> This is the **official and maintained** repository.  
> **DO NOT use** the old repository `FarininiChemometricSolutions/chemometricsolutions-demos` - it is no longer maintained.  
> 📧 Contact: chemometricsolutions@gmail.com

Professional chemometric analysis tools brought to the web. A comprehensive Streamlit-based platform for PCA, MLR/DoE, data handling, and classification with a fully modular architecture featuring root-level menu modules and shared workspace utilities.

**Live Demo:** https://chemometricsolutions-demo.streamlit.app/  
**GitHub:** https://github.com/EFarinini/chemometricsolutions-demo

---

**ChemometricSolutions** - Making Professional Chemometric Analysis Accessible to Everyone 🧪📊✨

---
## 📂 Project Structure

```
chemometricsolutions-demo/
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
- T² and Q statistics with control limits
- Varimax/Promax rotation options
- Sample contribution analysis

**Backend Connection:** `modules/pca/` (calculations.py, diagnostics.py, plots.py, statistics.py)

---

### **3. mlr_doe.py** - Multiple Linear Regression & Design of Experiments
**Entry Point:** `Main Menu → MLR & DoE`

**Features:**
- Full factorial design generation (2^k, 3^k)
- MLR model fitting with interaction terms
- Response surface visualization (3D plots)
- VIF analysis for multicollinearity
- Residual analysis and diagnostics
- Optimal point prediction

**Backend Connection:** `modules/mlr_doe/` (doe_generator.py, mlr_model.py, diagnostics.py, response_surface.py)

---

### **4. multi_doe_page.py** - Multi-Response DoE
**Entry Point:** `Main Menu → Multi-Response DoE`

**Features:**
- Multiple response optimization
- Desirability functions
- Pareto front visualization
- Trade-off analysis between responses

**Backend Connection:** `modules/mlr_doe/` (pareto_optimization.py, surface_analysis.py)

---

### **5. transformations.py** - Data Preprocessing
**Entry Point:** `Main Menu → Transformations`

**Features:**
- Centering (mean, median)
- Scaling (standardization, normalization, autoscaling)
- Spectral preprocessing (SNV, MSC, derivatives)
- Missing data handling
- Before/after visualization

**Backend Connection:** `modules/transformations/` (scaling.py, centering.py, spectral.py, missing_data.py)

---

### **6. pca_monitoring_page.py** - Quality Control & SPC
**Entry Point:** `Main Menu → Quality Control`

**Features:**
- PCA model training on reference data
- T² and Q control charts
- Real-time monitoring simulation
- Fault detection and diagnosis
- Contribution plots for out-of-control points

**Backend Connection:** `modules/quality_control/` (pca_monitoring.py, control_charts.py, fault_detection.py)

---

### **7. bayesian_optimization_page.py** - Bayesian Optimization
**Entry Point:** `Main Menu → Bayesian Optimization`

**Features:**
- Gaussian Process surrogate model
- Acquisition function visualization (EI, UCB, POI)
- Sequential experimental design
- Convergence analysis

**Backend Connection:** `modules/bayesian_optimization/` (gaussian_process.py, acquisition.py, optimization.py)

---

### **8. classification_page.py** - Classification Methods
**Entry Point:** `Main Menu → Classification`

**Features:**
- PLS-DA (Partial Least Squares Discriminant Analysis)
- SIMCA (Soft Independent Modeling of Class Analogy)
- LDA (Linear Discriminant Analysis)
- KNN (K-Nearest Neighbors)
- Confusion matrix and ROC curves
- Cross-validation metrics

**Backend Connection:** `modules/classification/` (models.py, training.py, evaluation.py, plots.py)

---

### **9. calibration_page.py** - PLS Calibration
**Entry Point:** `Main Menu → Calibration`

**Features:**
- PLS1/PLS2 regression
- Latent variable selection (cross-validation)
- Prediction with uncertainty
- Model diagnostics (R², RMSEC, RMSECV, RMSEP)
- Leverage and influence analysis

**Backend Connection:** `modules/calibration/` (pls_regression.py, calibration.py, predictions.py, diagnostics.py)

---

### **10. univariate_page.py** - Univariate Statistics
**Entry Point:** `Main Menu → Univariate`

**Features:**
- Descriptive statistics
- Hypothesis testing (t-test, ANOVA)
- Distribution fitting
- Correlation analysis
- Outlier detection

**Backend Connection:** `modules/univariate/` (descriptive_stats.py, hypothesis_tests.py, distributions.py)

---

## 🎨 Color Utilities

### **color_utils.py** (Root-level)

Provides unified color palettes for all visualizations:

```python
from color_utils import get_color_palette, get_theme_colors

# Get categorical color palette
colors = get_color_palette('categorical', n_colors=10)

# Get theme-specific colors (dark/light mode)
theme = get_theme_colors(dark_mode=True)
```

**Available Palettes:**
- `categorical` - For discrete groups
- `sequential` - For continuous values
- `diverging` - For values around a midpoint
- `qualitative` - High-contrast categorical

---

## 🔌 Visualization Module

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
# Clone repository (OFFICIAL REPOSITORY)
git clone https://github.com/EFarinini/chemometricsolutions-demo.git
cd chemometricsolutions-demo

# Create virtual environment
python3 -m venv venv
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
- GitHub: https://github.com/EFarinini
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
