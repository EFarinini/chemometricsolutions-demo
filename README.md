# ChemometricSolutions
Professional chemometric analysis on the web. Modular Streamlit app with PCA, MLR/DoE, classification, calibration, and more.

**Live Demo:** https://chemometricsolutions-demo.streamlit.app/

---

## 🏗️ Architecture

```
root/
├── streamlit_app.py              # Main entry
├── homepage.py                   # Navigation & dashboard
├── data_handling.py              # Data I/O
├── pca.py                        # PCA analysis
├── mlr_doe.py                    # MLR & DoE
├── multi_doe_page.py             # Multi-response DoE
├── transformations.py            # Data preprocessing
├── pca_monitoring_page.py        # Quality control
├── classification_page.py        # Classification
├── calibration_page.py           # PLS calibration
├── univariate_page.py            # Univariate stats
├── bivariate_page.py             # Bivariate analysis
├── generate_doe.py               # DoE generator
├── mixture_design.py             # Mixture designs
├── ga_variable_selection_page.py # GA variable selection
│
├── 🔧 Common Utilities (Root)
│   ├── color_utils.py            # Color palettes & themes
│   ├── workspace_utils.py        # Shared dataset workspace
│   ├── auth_utils.py             # Authentication
│   └── session_state_keys.py     # Session state keys
│
└── 📁 modules/                   # Calculation engines
    ├── data_handling/
    ├── pca/
    ├── mlr_doe/
    ├── transformations/
    ├── quality_control/
    ├── classification/
    ├── calibration/
    ├── univariate/
    └── visualization/
```

---

## 💡 Key Principles

✅ **Separation of Concerns:** Calculations in `modules/`, UI in root `.py` files  
✅ **Shared Workspace:** Single dataset loaded, accessible everywhere via `workspace_utils`  
✅ **Modular:** Each module folder works standalone  
✅ **Reusable:** Calculation functions can be imported and used anywhere  

---

## 🔗 Workspace System

Load data **once** in Data Handling → Access **everywhere** via `workspace_utils`:

```python
from workspace_utils import get_current_dataset, activate_dataset_in_workspace

# Get current active dataset
data = get_current_dataset()

# Switch datasets
datasets = get_workspace_datasets()
activate_dataset_in_workspace("my_dataset", datasets["my_dataset"])
```

**Flow:** Data Handling → workspace → Every module accesses via `get_current_dataset()`

---

## 📚 Stack

- **Framework:** Streamlit 1.28+
- **Compute:** NumPy, SciPy, scikit-learn
- **Data:** Pandas
- **Plots:** Plotly, Matplotlib
- **Deployment:** Streamlit Cloud

---

## 🤝 Contributing

1. Fork repo
2. Create feature branch
3. **Separate calculation logic from UI**
4. Update README
5. Submit PR

---

## 📄 License

MIT License - See LICENSE file

---

## 👨‍🔬 Author

**Dr. Emanuele Farinini, PhD**  
Chemometrics & Analytical Chemistry Expert

- Website: https://chemometricsolutions.com
- GitHub: https://github.com/FarininiChemometricSolutions
- Email: chemometricsolutions@gmail.com