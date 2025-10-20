# 🧪 ChemometricSolutions Interactive Demos

**Advanced chemometric tools for data analysis and multivariate statistics**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chemometricsolutions.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> Interactive Streamlit demos for chemometrics - PCA analysis, DoE simulator, process monitoring, and calibration tools.

---

## 🚀 Live Demo

**Try it now:** [chemometricsolutions-demo.streamlit.app](https://chemometricsolutions-demo.streamlit.app)

---

## 📊 Featured Demos

### 1. 📊 Data Handling
Complete data import/export suite supporting multiple formats for analytical chemistry:

**Supported Formats:**
- **Spectroscopy**: RAW (XRD), DAT, ASC, SPC, JDX/DX
- **Standard**: CSV, TXT, Excel (XLS/XLSX), JSON
- **Advanced**: PRN, TSV, ARFF, ODS, HDF5, MAT

**Features:**
- Multi-format file import with auto-detection
- Spectroscopic data conversion (perfect for SAM to Excel)
- Data transformation and preprocessing
- Workspace management
- Metadata handling and variable classification
- Export to multiple formats

### 2. 🎯 PCA Analysis
Comprehensive Principal Component Analysis suite with professional algorithms:

**Analysis Methods:**
- Standard PCA (Principal Component Analysis)
- Varimax Rotation for enhanced interpretability

**Visualization Tools:**
- Interactive scores plots (2D/3D)
- Loadings plots with variable contributions
- Scree plots and variance analysis
- Convex hull visualization for group separation
- Dark/Light mode color schemes

**Diagnostics:**
- T² (Hotelling's T²) statistic
- Q (SPE) statistic for outlier detection
- Control limits at multiple confidence levels
- Leverage analysis
- Cross-validation metrics (Q², RMSECV)

**Advanced Features:**
- Sample selection by coordinates or manual input
- Dataset splitting and workspace management
- Custom variable creation (time trends, groups)
- Unified color mapping system
- Export results in multiple formats

### 3. 🧪 MLR & Design of Experiments
Multiple Linear Regression and DoE tools for experimental optimization:

**Capabilities:**
- Full factorial design generation
- Candidate points creation
- MLR model computation with diagnostics
- Interaction and quadratic terms
- Response surface visualization
- VIF (Variance Inflation Factors) analysis
- Confidence interval analysis
- Cross-validation (Leave-One-Out)

**Diagnostic Plots:**
- Experimental vs Fitted values
- Residuals analysis
- Coefficients bar plots with significance
- Leverage plots
- CV predictions

---

## 🎨 Key Features

- **🎯 Interactive Visualizations**: Real-time plots with Plotly and Chart.js
- **📁 Multi-Format Support**: Import/export CSV, Excel, spectroscopy data
- **🔬 Professional Algorithms**: Industry-standard chemometric methods
- **💾 Export Capabilities**: Download results in CSV, Excel, PDF formats
- **🎓 Educational**: Learn chemometrics through hands-on demonstrations
- **🌙 Dark Mode**: Professional color schemes for presentations
- **📊 Advanced Diagnostics**: T², Q statistics, control charts
- **🔄 Workspace Management**: Save and reload analysis sessions

---

## 🛠️ Local Installation

```bash
# Clone the repository
git clone https://github.com/FarininiChemometricSolutions/chemometricsolutions-demos.git
cd chemometricsolutions-demos

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📚 Documentation

### Getting Started

1. **Load Data**: Go to "Data Handling" tab and upload your dataset
2. **Explore**: Use PCA Analysis for multivariate exploration
3. **Analyze**: Generate diagnostic plots and statistics
4. **Export**: Download results for reports and presentations

### Data Handling Module

**Import workflows:**
- Upload files in supported formats
- Auto-detect format and encoding
- Preview data before processing
- Handle missing values
- Transpose spectral data matrices

**Export workflows:**
- Convert between formats (e.g., SAM → Excel)
- Export with custom settings
- Save workspace for later analysis

### PCA Analysis Workflow

1. **Model Computation**: 
   - Select variables and samples
   - Choose Standard PCA or Varimax rotation
   - Configure preprocessing (centering, scaling)

2. **Variance Analysis**:
   - Scree plots
   - Cumulative variance plots
   - Variable contribution analysis
   - Random data comparison

3. **Scores Visualization**:
   - 2D and 3D plots
   - Color by categorical variables
   - Add convex hulls for groups
   - Sample selection and splitting

4. **Diagnostics**:
   - T² vs Q plots
   - Control limit analysis
   - Outlier detection
   - Time series monitoring

5. **Export Results**:
   - Scores and loadings
   - Variance summaries
   - Complete analysis reports

### MLR/DoE Workflow

1. **Generate candidate points** for experimental design
2. **Fit MLR model** with interactions and quadratic terms
3. **Analyze diagnostics** (R², VIF, leverage)
4. **Visualize results** (coefficients, residuals)
5. **Make predictions** for new experimental conditions

---

## 🧬 Use Cases

### Pharmaceutical Industry
- NIR spectroscopy for quality control
- Active ingredient quantification
- Batch release testing
- Process analytical technology (PAT)

### Chemical Process Monitoring
- Real-time multivariate analysis
- Process optimization with DoE
- Fault detection and diagnosis
- Quality by Design (QbD)

### Materials Science
- XRD pattern analysis
- Composition characterization
- Property prediction
- Materials discovery

### Environmental Chemistry
- Complex mixture analysis
- Contaminant detection
- Water quality monitoring
- Source apportionment

### Food Science
- Composition analysis
- Authentication studies
- Quality assessment
- Shelf-life prediction

---

## 🤝 About

Developed by **Dr. Emanuele Farinini, PhD**  
*Chemometric Consultant & Data Analysis Expert*

These interactive demonstrations showcase the capabilities of ChemometricSolutions software and methodologies. Each demo is designed to demonstrate real-world applications of chemometric methods with the same analytical approaches and algorithms used in commercial software solutions.

**Perfect for:**
- Method evaluation and validation
- Training and education
- Proof of concept studies
- Data exploration and visualization
- Client demonstrations

---

## 📧 Contact & Links

- **Website**: [chemometricsolutions.com](https://chemometricsolutions.com)
- **GitHub**: [github.com/FarininiChemometricSolutions](https://github.com/FarininiChemometricSolutions)
- **CAT Software**: [gruppochemiometria.it/software](https://gruppochemiometria.it/index.php/software)
- **LinkedIn**: Connect for custom solutions and consulting

---

## 🔧 Technical Stack

Built with modern Python libraries:

- **[Streamlit](https://streamlit.io)** - Interactive web application framework
- **[Plotly](https://plotly.com)** - Interactive visualizations and charts
- **[scikit-learn](https://scikit-learn.org)** - Machine learning algorithms (PCA, regression)
- **[Pandas](https://pandas.pydata.org)** - Data manipulation and analysis
- **[NumPy](https://numpy.org)** - Numerical computing
- **[SciPy](https://scipy.org)** - Scientific computing and statistics

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 FarininiChemometricSolutions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🔄 Version History

### v1.0.0 (January 2025) - Initial Release
- ✅ Data Handling module with multi-format support
- ✅ Complete PCA Analysis suite
- ✅ Standard PCA and Varimax rotation
- ✅ Advanced diagnostics (T²/Q statistics)
- ✅ Unified color system for consistent theming
- ✅ Workspace management and dataset splitting
- ✅ MLR and DoE module (beta)

### Planned Features
- 🔜 Response surface methodology (RSM)
- 🔜 Partial Least Squares (PLS) regression
- 🔜 Classification methods (PLS-DA, SIMCA)
- 🔜 Preprocessing methods (SNV, MSC, derivatives)
- 🔜 Batch processing capabilities
- 🔜 Advanced reporting with PDF export

---

## 🙏 Acknowledgments

Special thanks to:
- The open-source community for excellent tools and libraries
- Chemometrics researchers for methodology development
- Users and beta testers for valuable feedback

---

## 📊 Project Statistics

- **Languages**: Python, HTML, CSS
- **Primary Language**: Python (100%)
- **Modules**: 7 core modules
- **Functions**: 150+ analysis functions
- **Supported Formats**: 15+ file formats
- **Visualization Types**: 20+ plot types

---

## 🎓 Educational Resources

This project is ideal for:
- **University courses** in chemometrics and analytical chemistry
- **Industry training** for quality control and process monitoring
- **Self-learning** multivariate data analysis
- **Research** method development and validation

---

## 🤝 Contributing

We welcome contributions! Areas for contribution:
- Additional file format support
- New chemometric methods
- Improved visualizations
- Bug fixes and optimizations
- Documentation improvements

Please open an issue or pull request on GitHub.

---

## ⭐ Support This Project

If you find this project useful:
- **Star** this repository on GitHub
- **Share** with colleagues and students
- **Cite** in your research or publications
- **Provide feedback** for improvements

---

## 📞 Need Custom Solutions?

These demos represent a subset of our full chemometric capabilities. For:
- **Custom algorithm development**
- **Enterprise solutions**
- **Consulting services**
- **Training and workshops**
- **Method validation**

Contact us for tailored solutions that meet your specific needs.

---

**Made with ❤️ for the chemometrics community**

*Empowering scientists and analysts with professional data analysis tools*

---

## 🔗 Quick Links

- [📺 Live Demo](https://chemometricsolutions.streamlit.app)
- [📖 Documentation](https://github.com/FarininiChemometricSolutions/chemometricsolutions-demos)
- [🐛 Report Issues](https://github.com/FarininiChemometricSolutions/chemometricsolutions-demos/issues)
- [💡 Request Features](https://github.com/FarininiChemometricSolutions/chemometricsolutions-demos/issues)
- [📧 Contact](mailto:info@chemometricsolutions.com)

---

**Last Updated**: January 2025  
**Status**: Active Development  
**Deployment**: Streamlit Cloud