# PCA Quality Control - Pretreatment Tracking Implementation

## Overview

This document describes the implementation of **automatic pretreatment tracking and application** in the PCA Quality Control module. This feature ensures proper model validation by applying the exact same preprocessing to test data using training set statistics.

**Implementation Date**: 2025-10-22
**Author**: ChemometricSolutions

---

## Problem Statement

### The Challenge

When building PCA models for quality control, data is often pretreated (e.g., SNV, derivatives, autoscaling) to improve model performance. However, **test data must be preprocessed using the same transformations with training set statistics** - not recalculated on the test set.

**Why This Matters**:
- Using test set statistics for preprocessing causes **data leakage** and invalidates the model
- Test samples should be treated as "new" data with no prior knowledge
- Training set mean, std, min, max, etc. must be used for test data preprocessing

### Before This Implementation

- Model training used transformed data but didn't track which pretreatments were applied
- Test data selection didn't automatically apply the same pretreatments
- Users had to manually ensure test data was preprocessed identically
- Risk of incorrect results due to preprocessing mismatches

---

## Solution Architecture

### Modular Design

The solution is implemented as a **reusable, modular system** that can be extended to other sections of the application.

### Key Components

1. **PretreatmentTracker Class** (`pca_utils/pca_pretreatments.py`)
   - Detects which pretreatments have been applied to a dataset
   - Saves training set statistics (mean, std, min, max, etc.)
   - Applies saved pretreatments to test data using training statistics
   - Serializable for model persistence

2. **Model Training Integration** (`pca_monitoring_page.py` - Tab 1)
   - Automatically detects pretreatments from `transformation_history`
   - Saves pretreatment statistics when training the PCA model
   - Stores the `PretreatmentTracker` with the model in session state

3. **Test Data Integration** (`pca_monitoring_page.py` - Tab 3)
   - Retrieves saved `PretreatmentTracker` from training
   - Automatically applies pretreatments to test data using training statistics
   - Provides clear user feedback about which pretreatments were applied

---

## Implementation Details

### 1. PretreatmentTracker Class

Located in: `pca_utils/pca_pretreatments.py`

#### Key Methods

```python
class PretreatmentTracker:
    def detect_pretreatments(dataset_name, transformation_history) -> bool
        """Detect which pretreatments have been applied to a dataset"""

    def save_training_statistics(training_data, col_range)
        """Calculate and save statistics from training data"""

    def apply_to_test_data(test_data) -> pd.DataFrame
        """Apply saved pretreatments to test data using training statistics"""

    def to_dict() -> dict
        """Serialize for storage"""

    @classmethod
    def from_dict(data) -> PretreatmentTracker
        """Deserialize from storage"""
```

#### Supported Pretreatments

The tracker supports the following transformations:

**Row Transformations** (applied across variables):
- **SNV (Standard Normal Variate)**: Per-sample normalization
- **Derivatives**: First/second derivatives (deterministic)
- **Savitzky-Golay**: Smoothing and derivatives
- **Moving Average**: Smoothing
- **Row Sum = 100**: Per-row normalization
- **Binning**: Variable reduction

**Column Transformations** (applied within variables):
- **Centering**: Subtract training mean
- **Autoscaling**: Subtract training mean, divide by training std
- **Scaling**: Divide by training std (no centering)
- **Range [0,1]**: Use training min/max
- **Range [-1,1] (DoE Coding)**: Use training min/max
- **Maximum = 100**: Use training max
- **Sum = 100**: Use training sum
- **Log**: Use training delta for negative values

### 2. Model Training Integration

Located in: `pca_monitoring_page.py` - Lines 876-966

#### What Was Added

1. **Pretreatment Detection Section** (before training button):
   ```python
   # Detect pretreatments from transformation history
   pretreatment_tracker = detect_and_create_tracker(
       current_dataset_name,
       train_data,
       st.session_state.get('transformation_history', {})
   )

   if pretreatment_tracker:
       display_pretreatment_info(pretreatment_tracker)
   ```

2. **Statistics Saving** (after PCA model training):
   ```python
   if pretreatment_tracker is not None:
       # Get original untransformed data
       original_train_data = get_original_data(...)

       # Save training statistics
       pretreatment_tracker.save_training_statistics(
           original_train_data,
           col_range_for_stats
       )

   # Store with model
   st.session_state.pca_monitor_pretreatment_tracker = pretreatment_tracker
   ```

#### User Feedback

The Model Training tab now shows:
- Which pretreatments were detected
- Pretreatment parameters
- Confirmation that statistics were saved
- Warning that pretreatments will be automatically applied to test data

### 3. Test Data Integration

Located in: `pca_monitoring_page.py` - Lines 1504-1558

#### What Was Added

1. **Automatic Pretreatment Application Section**:
   ```python
   # Check if pretreatment tracker exists from training
   pretreatment_tracker_test = st.session_state.get('pca_monitor_pretreatment_tracker')

   if pretreatment_tracker_test is not None:
       # Get ORIGINAL (untransformed) test data
       original_test_data = find_original_data(...)

       # Apply pretreatments using TRAINING statistics
       test_data_pretreated = pretreatment_tracker_test.apply_to_test_data(
           original_test_data
       )

       # Use pretreated data for testing
       X_test = test_data_pretreated[model_vars]
   ```

2. **Original Data Resolution**:
   - Checks if test dataset is in `transformation_history`
   - Traces back to find the original untransformed version
   - Falls back to current data if not found

#### User Feedback

The Testing & Monitoring tab now shows:
- Detection of pretreatments from training
- Which pretreatments are being applied
- Confirmation that training statistics are being used
- Success/error messages for preprocessing

---

## Workflow Example

### Complete Quality Control Workflow

1. **Data Transformation** (Transformations page):
   ```
   User loads NIR_Data.csv
   → Applies SNV transformation
   → Saved as NIR_Data.snv in transformation_history
   ```

2. **Model Training** (Quality Control - Model Training tab):
   ```
   User selects NIR_Data.snv as training data
   → System detects SNV pretreatment
   → Calculates training statistics (mean, std for each row)
   → Trains PCA model on transformed data
   → Saves PretreatmentTracker with training statistics
   → Stores tracker in session state

   UI shows:
   ✅ Pretreatments detected: 1 transformation(s)
      • SNV (Standard Normal Variate)
      • Type: row_transformation
   ✅ Pretreatment statistics saved from training data
   📊 Pretreatments and statistics saved - will be automatically applied to test datasets
   ```

3. **Testing** (Quality Control - Testing & Monitoring tab):
   ```
   User selects Test_Set.csv from workspace
   → System retrieves PretreatmentTracker from training
   → Detects that SNV was applied to training
   → Finds original (untransformed) Test_Set.csv
   → Applies SNV using TRAINING SET statistics
   → Projects pretreated test data onto PCA model

   UI shows:
   ⚠️ Pretreatments detected from training - applying automatically to test data...
   ✅ Pretreatments detected: 1 transformation(s)
      • SNV (Standard Normal Variate)
   📈 Training statistics saved for 1 pretreatment(s)
   ✅ Pretreatments applied successfully using training statistics!
   📊 Pretreated test data: 50 samples × 100 variables
   ```

4. **Results**:
   - Test data is preprocessed correctly using training statistics
   - T² and Q statistics are calculated accurately
   - Fault detection is valid
   - Contribution analysis shows correct variables

---

## Technical Considerations

### Training Statistics Storage

For each pretreatment type, the following statistics are saved:

| Pretreatment | Statistics Saved | Usage |
|-------------|-----------------|-------|
| Centering | `mean` per column | Subtract from test data |
| Autoscaling | `mean`, `std` per column | Center and scale test data |
| Range [0,1] | `min`, `max` per column | Apply same scaling to test |
| Range [-1,1] | `min`, `max` per column | Apply same scaling to test |
| Maximum = 100 | `max` per column | Scale test data |
| Sum = 100 | `sum` per column | Scale test data |
| Log | `delta` (for negative values) | Add same delta to test |
| SNV | None (per-sample operation) | Calculate per test sample |
| Derivatives | None (deterministic) | Apply same operation to test |

### Original Data Resolution

The system uses a smart algorithm to find the original (untransformed) test data:

1. Check if test dataset is in `transformation_history`
2. If yes, get its `original_dataset` field
3. Look for the original dataset in `transformation_history` or `current_data`
4. If not found in transformation history, assume test dataset IS the original
5. If all fails, use current test data with a warning

### Error Handling

The implementation includes comprehensive error handling:

- **Missing original data**: Warning message, continues with available data
- **Dimension mismatch**: Clear error message, prevents testing
- **Preprocessing errors**: Error message with traceback, continues with untransformed data
- **Unknown transformations**: Warning message, requires manual handling

---

## Future Enhancements

### Potential Improvements

1. **Multiple Pretreatment Chains**:
   - Currently supports single pretreatment (last in chain)
   - Could be extended to track full transformation chains
   - Example: SNV → Savitzky-Golay → Autoscaling

2. **Model Persistence**:
   - Serialize `PretreatmentTracker` with saved models
   - Load pretreatment info when loading saved models
   - Export/import functionality

3. **Pretreatment Comparison**:
   - Compare test data before/after pretreatment
   - Visualize the effect of applying training statistics
   - Show differences if test data is already pretreated

4. **Advanced Transformations**:
   - Support for more complex transformations
   - Block scaling with multiple blocks
   - Custom transformation functions

5. **Validation Checks**:
   - Verify test data is in similar range as training
   - Detect outliers in preprocessing (e.g., test values outside training min/max)
   - Warning if test data appears pre-transformed

### Extension to Other Modules

The `PretreatmentTracker` is designed to be reusable:

```python
from pca_utils.pca_pretreatments import PretreatmentTracker

# In any module that needs pretreatment tracking
tracker = PretreatmentTracker()
tracker.detect_pretreatments(dataset_name, transformation_history)
tracker.save_training_statistics(training_data, col_range)

# Later, apply to new data
preprocessed_data = tracker.apply_to_test_data(new_data)
```

Can be used in:
- MLR/DoE module
- Other supervised learning methods
- Batch process monitoring
- Model deployment

---

## Code Locations

### New Files

- **`pca_utils/pca_pretreatments.py`**: Complete pretreatment tracking module (480 lines)

### Modified Files

- **`pca_monitoring_page.py`**:
  - Import section: Added pretreatment module imports
  - Model Training tab (lines 876-966): Added pretreatment detection and statistics saving
  - Testing & Monitoring tab (lines 1504-1558): Added automatic pretreatment application

### Key Functions

```python
# pca_utils/pca_pretreatments.py
class PretreatmentTracker:
    detect_pretreatments()
    save_training_statistics()
    apply_to_test_data()
    get_summary()
    to_dict() / from_dict()

# Helper functions
detect_and_create_tracker()
display_pretreatment_info()
```

---

## Testing Recommendations

### Test Scenarios

1. **SNV Pretreatment**:
   - Transform training data with SNV
   - Train model
   - Test on untransformed test data → Should auto-apply SNV

2. **Autoscaling**:
   - Transform training data with autoscaling
   - Train model
   - Test on untransformed test data → Should use training mean/std

3. **Range [-1,1] (DoE)**:
   - Transform training data to [-1,1] range
   - Train model
   - Test on untransformed test data → Should use training min/max

4. **No Pretreatment**:
   - Train model on raw data
   - Test on raw data → Should work normally

5. **Derivative + Autoscaling** (future):
   - Apply multiple transformations
   - Verify full chain is tracked

### Validation Checks

- Verify test results match manual preprocessing
- Compare T²/Q values with reference implementation
- Check contribution analysis makes sense
- Ensure warnings appear for edge cases

---

## Summary

### What Was Accomplished

✅ **Created modular pretreatment tracking system** (`pca_utils/pca_pretreatments.py`)
✅ **Integrated into Model Training tab** - detects and saves pretreatment statistics
✅ **Integrated into Testing & Monitoring tab** - automatically applies pretreatments to test data
✅ **Comprehensive user feedback** - clear messages about what's happening
✅ **Error handling** - graceful degradation with warnings
✅ **Supports all major transformations** - row and column transformations
✅ **Reusable design** - can be extended to other modules

### Key Benefits

1. **Correctness**: Ensures test data is preprocessed correctly using training statistics
2. **Automation**: No manual preprocessing required for test data
3. **Transparency**: Clear feedback about which pretreatments are applied
4. **Modularity**: Reusable component for other analysis modules
5. **Future-proof**: Easy to extend with new transformation types

### User Impact

- **Before**: Users had to manually preprocess test data, risking errors
- **After**: System automatically applies correct preprocessing with training statistics
- **Result**: More accurate quality control, less room for error, better user experience

---

## Contact

For questions or issues related to this implementation:
- **Developer**: ChemometricSolutions
- **Date**: 2025-10-22
- **Module**: PCA Quality Control - Pretreatment Tracking
