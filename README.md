# ChemometricSolutions - Modular Web Application

Professional chemometric analysis tools brought to the web. A comprehensive Streamlit-based platform for PCA, MLR/DoE, data handling, and classification with a fully modular architecture.

**Live Demo:** https://chemometricsolutions-demo.streamlit.app/  
**GitHub:** https://github.com/FarininiChemometricSolutions/chemometricsolutions-demos

---

## 📂 Project Structure

```
chemometricsolutions-demos/
│
├── streamlit_app.py                 # Main entry point / Homepage
├── requirements.txt                 # Python dependencies
├── config.py                        # Global configuration
├── color_utils.py                   # Theme colors & branding
│
├── pages/                           # Streamlit multi-page app
│   ├── 1_Data_Handling.py          # Data import/export UI
│   ├── 2_PCA_Analysis.py           # PCA analysis UI
│   ├── 3_MLR_DoE.py                # MLR & Design of Experiments UI
│   └── 4_Classification.py         # Classification algorithms UI
│
└── modules/                         # Core calculation modules
    ├── __init__.py
    │
    ├── data_handling/              # Data I/O and management
    │   ├── __init__.py
    │   ├── loaders.py              # Load CSV, Excel, RAW files
    │   ├── exporters.py            # Export data in multiple formats
    │   ├── transformations.py      # Data transformations & operations
    │   ├── validators.py           # Input validation & error handling
    │   └── workspace_utils.py      # Workspace management
    │
    ├── pca/                        # Principal Component Analysis
    │   ├── __init__.py
    │   ├── calculations.py         # Core PCA, SVD, rotations (Varimax, Promax)
    │   ├── diagnostics.py          # T² & Q statistics, contributions
    │   ├── plots.py                # 2D/3D scores, loadings plots
    │   ├── statistics.py           # Variance explained, eigenvalues
    │   ├── monitoring.py           # PCA monitoring & control charts
    │   └── ai_utils.py             # AI-assisted diagnostics
    │
    ├── mlr_doe/                    # Multiple Linear Regression & DoE
    │   ├── __init__.py
    │   ├── doe_generator.py        # Generate factorial designs (2^k, 3^k)
    │   ├── mlr_model.py            # MLR model computation
    │   ├── diagnostics.py          # VIF, residuals, model quality
    │   ├── response_surface.py     # Response surface analysis
    │   ├── candidate_points.py     # Optimization candidate selection
    │   └── confidence_intervals.py # Confidence intervals & uncertainty
    │
    ├── preprocessing/              # Data preprocessing
    │   ├── __init__.py
    │   ├── scaling.py              # Standardization, normalization, autoscaling
    │   ├── centering.py            # Mean centering operations
    │   ├── spectral.py             # Spectral preprocessing (SNV, MSC, derivatives)
    │   ├── missing_data.py         # Missing value reconstruction
    │   ├── column_transforms.py    # Column-wise transformations
    │   ├── row_transforms.py       # Row-wise transformations
    │   └── transform_plots.py      # Visualization of transformations
    │
    ├── calibration/                # PLS Multivariate Calibration
    │   ├── __init__.py
    │   ├── pls_regression.py       # PLS model computation & prediction
    │   ├── calibration.py          # Calibration & cross-validation
    │   ├── predictions.py          # Sample predictions & uncertainty quantification
    │   └── diagnostics.py          # Model diagnostics & outlier detection
    │
    ├── classification/             # Classification & pattern recognition
    │   ├── __init__.py
    │   ├── models.py               # Classification algorithms (PLS-DA, SIMCA, etc.)
    │   ├── training.py             # Model training & cross-validation
    │   ├── evaluation.py           # Performance metrics, confusion matrix
    │   ├── plots.py                # Classification-specific visualizations
    │   └── diagnostics.py          # Model diagnostics
    │
    └── visualization/              # Unified visualization system
        ├── __init__.py
        ├── colors.py               # Color palettes & theme management
        ├── plots_common.py         # Shared plotting utilities
        └── themes.py               # Consistent plot styling
```

---

## 🎯 Core Modules

### 1. **Data Handling Module** (`modules/data_handling/`)
Manages all data import/export operations with support for multiple file formats.

**Key Files:**
- `loaders.py` - Load CSV, Excel (.xlsx, .xls), RAW spectral files
- `exporters.py` - Export processed data, backup datasets
- `transformations.py` - Row operations, column operations, filtering
- `validators.py` - Data validation, type checking, error handling

**Usage:**
```python
from modules.data_handling import loaders, exporters
data = loaders.load_csv("dataset.csv")
exporters.export_excel(data, "output.xlsx")
```

---

### 2. **PCA Module** (`modules/pca/`)
Complete Principal Component Analysis suite with diagnostics and visualizations.

**Key Files:**
- `calculations.py` - Standard PCA, SVD, Varimax rotation
- `diagnostics.py` - T² (Hotelling's), Q (SPE) statistics, contributions
- `plots.py` - 2D/3D score plots, loading plots, biplot
- `statistics.py` - Variance explained, cumulative variance, eigenvalues
- `monitoring.py` - PCA monitoring charts, control limits
- `ai_utils.py` - AI-powered diagnostics and anomaly detection

**Usage:**
```python
from modules.pca import calculations, plots
loadings, scores, variance = calculations.compute_pca(data, n_components=3)
plots.plot_scores_2d(scores, targets)
```

---

### 3. **MLR & DoE Module** (`modules/mlr_doe/`)
Multiple Linear Regression and Design of Experiments tools.

**Key Files:**
- `doe_generator.py` - Generate full factorial designs (2^k, 3^k, mixed)
- `mlr_model.py` - MLR computation, coefficients, model equations
- `diagnostics.py` - VIF (Variance Inflation Factor), residuals, R², RMSE
- `response_surface.py` - Response surface methodology, 3D visualization
- `candidate_points.py` - Optimal point selection for next experiments
- `confidence_intervals.py` - Prediction intervals, uncertainty quantification

**Usage:**
```python
from modules.mlr_doe import doe_generator, mlr_model
design = doe_generator.generate_factorial_design(factors=3, levels=2)
model = mlr_model.compute_mlr(X, y)
```

---

### 4. **Preprocessing Module** (`modules/preprocessing/`)
Data preprocessing and spectral transformation suite.

**Key Files:**
- `scaling.py` - Standardization (z-score), normalization, autoscaling
- `centering.py` - Mean centering, column centering
- `spectral.py` - SNV, MSC, 1st/2nd derivatives, Savitzky-Golay
- `missing_data.py` - Missing value reconstruction, imputation
- `column_transforms.py` - Log transform, square root, polynomial
- `row_transforms.py` - Row normalization, outlier detection
- `transform_plots.py` - Before/after transformation visualization

**Usage:**
```python
from modules.preprocessing import scaling, spectral
scaled_data = scaling.standardize(data)
pretreated = spectral.savitzky_golay(data, window=5, order=2)
```

---

### 5. **Calibration (PLS) Module** (`modules/calibration/`)
Partial Least Squares regression for quantitative multivariate calibration.

**Key Files:**
- `pls_regression.py` - PLS model computation, X/Y loadings & scores
- `calibration.py` - Model calibration, cross-validation, optimal LV selection
- `predictions.py` - Sample predictions, prediction intervals, uncertainty quantification
- `diagnostics.py` - Model quality metrics, outlier detection, leverage analysis

**Usage:**
```python
from modules.calibration import pls_regression, predictions
model = pls_regression.compute_pls(X_cal, y_cal, n_components=5)
y_pred, intervals = predictions.predict_samples(model, X_test)
```

---

### 6. **Classification Module** (`modules/classification/`)
Supervised classification and pattern recognition.

**Key Files:**
- `models.py` - PLS-DA, SIMCA, LDA, KNN classifiers
- `training.py` - Cross-validation, train/test split, hyperparameter tuning
- `evaluation.py` - Accuracy, precision, recall, F1-score, confusion matrix
- `plots.py` - Classification scores, class boundaries, ROC curves
- `diagnostics.py` - Feature importance, model reliability

**Usage:**
```python
from modules.classification import models, training, evaluation
clf = models.PLSDAClassifier(n_components=3)
scores = training.cross_validate(clf, X, y)
```

---

### 6. **Visualization Module** (`modules/visualization/`)
Unified visualization system ensuring consistent styling across all modules.

**Key Files:**
- `colors.py` - ChemometricSolutions color palette, theme management
- `plots_common.py` - Base plotting functions, grid layouts, common formatting
- `themes.py` - Plot styling, font settings, color schemes

**Usage:**
```python
from modules.visualization import colors, plots_common
theme = colors.get_chemometric_theme()
fig = plots_common.create_plotly_figure(theme)
```

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/FarininiChemometricSolutions/chemometricsolutions-demos.git
cd chemometricsolutions-demos

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

## 📊 Features

### Data Handling
- ✓ Multi-format support (CSV, Excel, RAW spectral files)
- ✓ Data validation and error checking
- ✓ Workspace management and data backups
- ✓ Data transformation and preprocessing pipelines

### PCA Analysis
- ✓ Standard PCA & Varimax rotation
- ✓ Interactive 2D/3D score plots
- ✓ Hotelling's T² & Q (SPE) statistics
- ✓ Loadings and biplot visualizations
- ✓ Variance explained analysis
- ✓ PCA monitoring and control charts
- ✓ AI-powered anomaly detection

### MLR & DoE
- ✓ Full factorial design generation (2^k, 3^k)
- ✓ MLR model computation with diagnostics
- ✓ VIF multicollinearity assessment
- ✓ Response surface methodology
- ✓ Optimal point candidate selection
- ✓ Confidence intervals and prediction intervals
- ✓ Model equation generation

### Preprocessing
- ✓ Standardization and normalization
- ✓ Mean centering
- ✓ Spectral preprocessing (SNV, MSC, derivatives)
- ✓ Missing data reconstruction
- ✓ Savitzky-Golay filtering
- ✓ Transformation visualization

### PLS Multivariate Calibration
- ✓ PLS1 & PLS2 regression models
- ✓ Cross-validation with optimal LV selection
- ✓ Prediction intervals and uncertainty quantification
- ✓ Model diagnostics (R², RMSEC, RMSECV, RMSEP)
- ✓ Outlier detection and leverage analysis
- ✓ Sample predictions with confidence intervals

### Classification
- ✓ PLS-DA, SIMCA, LDA, KNN classifiers
- ✓ Cross-validation and hyperparameter tuning
- ✓ Performance metrics (accuracy, precision, recall, F1)
- ✓ Confusion matrices and ROC curves
- ✓ Feature importance analysis

---

## 💻 Technology Stack

- **Backend:** Python 3.9+
  - NumPy, SciPy - Scientific computing
  - scikit-learn - Machine learning
  - pandas - Data manipulation
  
- **Frontend:** Streamlit
  - Interactive web interface
  - Real-time data visualization
  - Session state management
  
- **Visualization:** Plotly
  - Interactive 2D/3D plots
  - High-quality publication-ready figures
  - Responsive design
  
- **Deployment:** Streamlit Cloud
  - Free cloud hosting
  - Automatic updates from GitHub
  - Scalable infrastructure

---

## 📈 Architecture Benefits

✅ **Modularity** - Independent, testable modules  
✅ **Maintainability** - Easy to debug and update individual components  
✅ **Scalability** - Add new analysis tools without affecting existing code  
✅ **Reusability** - Modules can be imported and used in other projects  
✅ **Testability** - Each module can have dedicated unit tests  
✅ **Performance** - Optimized calculation modules separate from UI  

---

## 📝 Development Guidelines

### Adding a New Module

1. Create folder under `modules/` with clear name
2. Implement calculation functions (no Streamlit code!)
3. Add import to `modules/__init__.py`
4. Create Streamlit page in `pages/`
5. Update documentation

### Code Style

- Follow PEP 8 conventions
- Use type hints for functions
- Document with docstrings
- Keep calculation logic separate from UI code

### Testing

```bash
# Run tests
pytest tests/

# Check code coverage
pytest --cov=modules tests/
```

---

## 📚 Documentation

- **Module Documentation:** See individual `README.md` in each module
- **API Reference:** https://github.com/FarininiChemometricSolutions/chemometricsolutions-demos/wiki
- **Examples:** See `examples/` folder for sample workflows
- **Theory:** See `docs/theory.md` for mathematical background

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

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

- Built with ❤️ using Python, Streamlit, and Plotly
- Reference R packages: CAT software, chemometrics packages
- Scientific foundations from peer-reviewed analytical chemistry literature

---

## 📞 Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Email:** chemometricsolutions@gmail.com
- **Live Demo:** https://chemometricsolutions-demo.streamlit.app

---

**ChemometricSolutions** - Making Professional Chemometric Analysis Accessible to Everyone 🧪📊✨
