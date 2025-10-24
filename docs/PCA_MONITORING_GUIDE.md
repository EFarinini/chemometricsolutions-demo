# PCA Process Monitoring Guide

## Overview

The PCA Monitoring module provides comprehensive tools for **multivariate statistical process control (MSPC)** using Principal Component Analysis. It's designed for real-time monitoring of industrial processes, detecting faults, and diagnosing their root causes.

## Key Features

### 1. Model Training
- Train PCA models from normal operating condition (NOC) data
- Automatic determination of optimal components
- Multiple scaling options (auto, pareto, none)
- Comprehensive model validation

### 2. Statistical Monitoring
- **T² (Hotelling) Statistic**: Detects unusual patterns within the model space
- **Q (SPE) Statistic**: Detects deviations from the model structure
- **Multiple Control Limits**: 97.5%, 99.5%, 99.95% confidence levels
- Based on rigorous statistical distributions (F-distribution for T², chi-square for Q)

### 3. Fault Detection
- Real-time fault detection on new process data
- Classification of fault types (T² fault, Q fault, or both)
- Boolean fault indicators for easy filtering

### 4. Fault Diagnosis
- Variable contribution analysis for both T² and Q statistics
- Interactive contribution plots showing which variables caused the fault
- Top-N contributor identification

### 5. Model Persistence
- Save trained models to disk (pickle format)
- Load models for deployment in production systems
- Complete state preservation including control limits

### 6. Visualization
- Interactive monitoring charts (Plotly-based)
- T² and Q time series with control limit bands
- Combined T² vs Q scatter plots for fault type visualization
- Variable contribution bar charts for diagnosis

---

## Installation

The module is part of the `pca_utils` package. Ensure you have the required dependencies:

```bash
pip install numpy pandas scipy scikit-learn plotly
```

---

## Quick Start

### Basic Workflow

```python
from pca_utils import PCAMonitor
import pandas as pd

# 1. Load your training data (normal operating conditions)
X_train = pd.read_csv('normal_data.csv')

# 2. Create and train the monitor
monitor = PCAMonitor(n_components=5, scaling='auto')
monitor.fit(X_train)

# 3. Test new data
X_test = pd.read_csv('new_data.csv')
results = monitor.predict(X_test)

# 4. Analyze results
print(f"Faults detected: {results['faults'].sum()}")

# 5. Create monitoring chart
fig = monitor.plot_monitoring_chart(results)
fig.show()
```

---

## Detailed Usage

### 1. Creating a PCA Monitor

```python
monitor = PCAMonitor(
    n_components=5,              # Number of PCs (int or None for automatic)
    scaling='auto',              # Scaling method: 'auto', 'pareto', 'none'
    alpha_levels=[0.975, 0.995, 0.9995]  # Control limit confidence levels
)
```

**Parameters:**
- `n_components`:
  - `int`: Use that many components
  - `float` (0-1): Use components explaining that fraction of variance
  - `None`: Use all components
- `scaling`:
  - `'auto'`: Mean centering + standardization (recommended)
  - `'pareto'`: Mean centering + Pareto scaling (sqrt of std dev)
  - `'none'`: No scaling (use if data is already scaled)
- `alpha_levels`: List of confidence levels for control limits

### 2. Training the Model

```python
# From pandas DataFrame
monitor.fit(X_train, feature_names=X_train.columns)

# From numpy array
monitor.fit(X_train_array, feature_names=['Temp1', 'Temp2', 'Pressure', ...])
```

**What happens during training:**
1. Data is scaled according to the specified method
2. PCA model is fitted
3. T² and Q control limits are calculated for each alpha level
4. Model parameters are stored for prediction

### 3. Predicting on New Data

```python
results = monitor.predict(X_test, return_contributions=True)
```

**Returns dictionary with:**
- `t2`: T² statistic for each sample
- `q`: Q statistic for each sample
- `t2_limits`: Dictionary of T² limits {alpha: limit}
- `q_limits`: Dictionary of Q limits {alpha: limit}
- `faults`: Boolean array indicating fault detection
- `fault_type`: Classification ('none', 't2', 'q', 'both')
- `scores`: PCA scores
- `X_scaled`: Scaled input data
- `contributions_t2`: T² contributions per variable (if requested)
- `contributions_q`: Q contributions per variable (if requested)

### 4. Analyzing Results

#### Get Fault Summary
```python
summary = monitor.get_fault_summary(results)
print(summary)
```

Output:
```
   Sample  T2_Statistic  Q_Statistic  Fault_Detected Fault_Type  T2_Exceeds_97.5%
0       1         5.234        0.456           False       none             False
1       2        15.678        2.345            True         t2              True
2       3         3.456       12.890            True          q             False
3       4        18.234       15.678            True       both              True
...
```

#### Identify Top Contributing Variables
```python
# For a specific faulty sample
sample_idx = 10
q_contrib = results['contributions_q'][sample_idx]

# Get top 5 contributors
top_5_idx = np.argsort(q_contrib)[-5:][::-1]
for idx in top_5_idx:
    var_name = monitor.feature_names_[idx]
    contrib = q_contrib[idx]
    print(f"{var_name}: {contrib:.3f}")
```

### 5. Visualization

#### Monitoring Chart
```python
# Time series of T² and Q with control limits
fig = monitor.plot_monitoring_chart(
    results,
    sample_labels=['2025-01-01 10:00', '2025-01-01 10:15', ...],
    title="Process Monitoring - January 2025"
)
fig.show()
# Or save to file
fig.write_html("monitoring_chart.html")
```

#### Contribution Plot
```python
# Show which variables contributed to a fault
fig = monitor.plot_contribution_chart(
    results,
    sample_idx=10,      # Sample to analyze
    statistic='q',      # 'q' or 't2'
    top_n=15           # Number of top contributors to show
)
fig.show()
```

#### T² vs Q Scatter Plot
```python
from pca_utils import plot_combined_monitoring_chart

fig = plot_combined_monitoring_chart(
    results,
    results['t2_limits'],
    results['q_limits'],
    sample_labels=sample_ids,
    title="Fault Detection Map"
)
fig.show()
```

### 6. Save and Load Models

#### Save Model
```python
# After training
monitor.save('models/process_monitor_v1.pkl')
```

#### Load Model
```python
# In production code
monitor = PCAMonitor.load('models/process_monitor_v1.pkl')

# Ready to use
results = monitor.predict(new_data)
```

### 7. Model Information

```python
# Get model summary
summary = monitor.get_model_summary()

print(f"Components: {summary['n_components']}")
print(f"Variance explained: {summary['variance_explained']*100:.2f}%")
print(f"T² limits: {summary['t2_limits']}")
print(f"Q limits: {summary['q_limits']}")
```

---

## Understanding the Statistics

### T² Statistic (Hotelling)

**What it measures:** Distance from the center of the PCA model in the retained component space.

**Formula:**
```
T² = Σ (score_i² / eigenvalue_i)
```

**Interpretation:**
- **Low T²**: Sample is close to the normal operating center
- **High T²**: Sample has unusual patterns in the measured variables, but still fits the correlation structure
- **Exceeds limit**: The sample is significantly different from normal conditions

**Typical causes:**
- Process operating point shift
- Grade transitions
- Controlled changes in operating conditions

### Q Statistic (SPE - Squared Prediction Error)

**What it measures:** Distance from the PCA model hyperplane (residual variance).

**Formula:**
```
Q = Σ (x_i - x_reconstructed_i)²
```

**Interpretation:**
- **Low Q**: Sample is well-described by the PCA model
- **High Q**: Sample doesn't follow the normal correlation structure
- **Exceeds limit**: New patterns or relationships not seen during training

**Typical causes:**
- Sensor failures or drift
- New types of disturbances
- Broken correlations between variables
- Process structural changes

### Fault Type Classification

| T² | Q | Fault Type | Typical Cause |
|---|---|---|---|
| Normal | Normal | `none` | Normal operation |
| High | Normal | `t2` | Operating point shift, controlled change |
| Normal | High | `q` | Sensor fault, new disturbance pattern |
| High | High | `both` | Major fault, multiple issues |

---

## Control Limits

### Multiple Confidence Levels

The module calculates control limits at three confidence levels by default:

- **97.5%** (α = 0.975): ~2.5% false alarm rate (1 in 40 samples)
- **99.5%** (α = 0.995): ~0.5% false alarm rate (1 in 200 samples)
- **99.95%** (α = 0.9995): ~0.05% false alarm rate (1 in 2000 samples)

**Choosing the right level:**
- **97.5%**: Sensitive detection, good for early warnings, higher false alarm rate
- **99.5%**: Balanced approach (recommended for most applications)
- **99.95%**: Very conservative, detects only severe faults, minimal false alarms

### Custom Confidence Levels

```python
monitor = PCAMonitor(
    n_components=5,
    alpha_levels=[0.95, 0.975, 0.99, 0.995, 0.999, 0.9995]
)
```

---

## Complete Example

```python
from pca_utils import PCAMonitor, plot_combined_monitoring_chart
import pandas as pd
import numpy as np

# Load training data (normal operating conditions)
train_data = pd.read_csv('normal_operation_2024.csv')
print(f"Training data: {train_data.shape}")

# Create monitor
monitor = PCAMonitor(
    n_components=0.95,  # Explain 95% of variance
    scaling='auto',
    alpha_levels=[0.975, 0.995, 0.9995]
)

# Train model
print("Training model...")
monitor.fit(train_data)

# Show model info
summary = monitor.get_model_summary()
print(f"\nModel trained with {summary['n_components']} components")
print(f"Variance explained: {summary['variance_explained']*100:.2f}%")

# Save model
monitor.save('production_monitor_v1.pkl')
print("Model saved!")

# Load test data
test_data = pd.read_csv('operation_jan_2025.csv')
print(f"\nTesting on {len(test_data)} new samples...")

# Predict
results = monitor.predict(test_data, return_contributions=True)

# Analyze faults
n_faults = results['faults'].sum()
print(f"\nFaults detected: {n_faults} ({n_faults/len(test_data)*100:.1f}%)")

# Get detailed summary
fault_summary = monitor.get_fault_summary(results)
faulty_samples = fault_summary[fault_summary['Fault_Detected']]
print(f"\nFaulty samples:\n{faulty_samples}")

# Create monitoring chart
fig1 = monitor.plot_monitoring_chart(
    results,
    sample_labels=test_data['timestamp'],
    title="Process Monitoring - January 2025"
)
fig1.write_html("monitoring_jan2025.html")

# Create T² vs Q chart
fig2 = plot_combined_monitoring_chart(
    results,
    results['t2_limits'],
    results['q_limits'],
    title="Fault Detection Map - January 2025"
)
fig2.write_html("t2_vs_q_jan2025.html")

# Diagnose first fault
if n_faults > 0:
    first_fault_idx = np.where(results['faults'])[0][0]
    print(f"\n--- Diagnosing first fault (sample {first_fault_idx+1}) ---")

    # Show statistics
    print(f"T² = {results['t2'][first_fault_idx]:.2f}")
    print(f"Q = {results['q'][first_fault_idx]:.2f}")
    print(f"Type = {results['fault_type'][first_fault_idx]}")

    # Show top contributors
    q_contrib = results['contributions_q'][first_fault_idx]
    top_5_idx = np.argsort(q_contrib)[-5:][::-1]

    print("\nTop 5 contributing variables:")
    for rank, idx in enumerate(top_5_idx, 1):
        var_name = monitor.feature_names_[idx]
        contrib = q_contrib[idx]
        print(f"  {rank}. {var_name}: {contrib:.3f}")

    # Create contribution plot
    fig3 = monitor.plot_contribution_chart(
        results,
        sample_idx=first_fault_idx,
        statistic='q',
        top_n=15
    )
    fig3.write_html(f"contribution_sample_{first_fault_idx+1}.html")

print("\nAnalysis complete! Check HTML files for interactive charts.")
```

---

## Best Practices

### 1. Training Data Selection
- Use data from **normal operating conditions only**
- Include sufficient variation in normal operation
- Minimum 100-200 samples recommended
- More samples than variables (n > p)
- Remove known faults and startup/shutdown periods

### 2. Number of Components
- Use cross-validation or scree plot to select
- Typical: explain 80-95% of variance
- Too few: Miss important patterns
- Too many: Model noise, reduced fault detection

### 3. Data Scaling
- **Always scale** unless data is already preprocessed
- Use `'auto'` (standardization) for most cases
- Use `'pareto'` for spectroscopic data
- Be consistent between training and testing

### 4. Control Limit Selection
- Start with **99.5%** for most applications
- Adjust based on false alarm tolerance
- Higher confidence = fewer false alarms but might miss subtle faults
- Monitor multiple levels during validation

### 5. Model Maintenance
- Retrain periodically as process evolves
- Monitor false alarm rates
- Update when process changes significantly
- Version your models

### 6. Fault Diagnosis
- Always check **both** T² and Q when investigating faults
- Use contribution plots to identify root causes
- Contributions show **which** variables, not necessarily **why**
- Combine with process knowledge for diagnosis

---

## Troubleshooting

### Issue: Too Many False Alarms

**Solutions:**
- Increase confidence level (e.g., 99.5% → 99.95%)
- Check if training data truly represents normal operation
- Increase number of training samples
- Reduce number of components (might be modeling noise)

### Issue: Missing Known Faults

**Solutions:**
- Decrease confidence level (e.g., 99.5% → 97.5%)
- Check if fault is in model space (T²) or residual space (Q)
- Increase number of components
- Verify scaling is consistent

### Issue: All Samples Flagged as Faulty

**Solutions:**
- Check data scaling matches between train and test
- Verify test data is from same process
- Check for data loading errors
- Ensure model was fitted before prediction

### Issue: Control Limits Are Very Large/Small

**Solutions:**
- Check training sample size (need n > p)
- Verify data quality (no outliers in training)
- Check scaling settings
- Ensure numerical stability (no zero variance variables)

---

## API Reference

### PCAMonitor Class

#### Constructor
```python
PCAMonitor(n_components=None, scaling='auto', alpha_levels=[0.975, 0.995, 0.9995])
```

#### Methods

| Method | Description |
|--------|-------------|
| `fit(X, feature_names)` | Train model on normal data |
| `predict(X, return_contributions)` | Test new data, return fault detection results |
| `plot_monitoring_chart(results, ...)` | Create T²/Q time series chart |
| `plot_contribution_chart(results, sample_idx, ...)` | Create contribution plot |
| `get_fault_summary(results)` | Generate fault summary DataFrame |
| `get_model_summary()` | Get model parameters and limits |
| `save(filepath)` | Save model to file |
| `load(filepath)` | Load model from file (classmethod) |

#### Attributes (after fitting)

| Attribute | Description |
|-----------|-------------|
| `pca_model_` | Fitted scikit-learn PCA model |
| `scaler_` | Fitted StandardScaler |
| `t2_limits_` | Dict of T² control limits |
| `q_limits_` | Dict of Q control limits |
| `loadings_` | PCA loadings matrix |
| `explained_variance_` | Variance explained by each PC |
| `feature_names_` | List of variable names |
| `n_components` | Number of components used |
| `is_fitted_` | Boolean indicating if model is fitted |

---

## References

### Scientific Literature

1. **Jackson, J.E. (1991)**. *A User's Guide to Principal Components*. Wiley.
   - Classic reference on PCA methodology

2. **Nomikos & MacGregor (1995)**. *Multivariate SPC charts for monitoring batch processes*. Technometrics, 37(1), 41-59.
   - Foundation of PCA-based process monitoring

3. **Kourti & MacGregor (1995)**. *Process analysis, monitoring and diagnosis using multivariate projection methods*. Chemometrics and Intelligent Laboratory Systems, 28(1), 3-21.
   - Comprehensive review of MSPC methods

4. **Jackson & Mudholkar (1979)**. *Control procedures for residuals associated with principal component analysis*. Technometrics, 21(3), 341-349.
   - Q statistic control limit methodology

5. **Wise et al. (2006)**. *Chemometrics with PCA*.
   - Practical guide to PCA in industrial applications

### Online Resources

- [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Plotly Python Documentation](https://plotly.com/python/)

---

## Support

For issues, questions, or contributions:
- Check the `example_pca_monitoring.py` script for working examples
- Review this guide for common use cases
- Examine the docstrings in `pca_monitoring.py` for detailed API documentation

---

## License

Part of ChemometricSolutions package
Copyright 2025
