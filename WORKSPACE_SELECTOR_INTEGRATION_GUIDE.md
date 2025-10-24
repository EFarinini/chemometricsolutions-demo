# Workspace Selector - Integration Guide for New Modules

## Quick Reference

This guide shows how to add the workspace dataset selector to **any new analysis module** in 3 simple steps.

---

## Standard Integration Pattern

### Step 1: Add Import

At the top of your module file, add:

```python
# Import workspace utilities for dataset selection
from workspace_utils import display_workspace_dataset_selector
```

### Step 2: Add Selector in show() Function

Replace the old data loading pattern with:

```python
def show():
    """Display the [Module Name] page"""

    st.markdown("# Module Title")
    st.markdown("*Module description*")

    # Dataset selection from workspace
    st.markdown("### 📊 Select Dataset for Analysis")

    result = display_workspace_dataset_selector(
        label="Select dataset from workspace:",
        key="unique_module_key_selector",  # MUST be unique per module!
        help_text="Choose a dataset to analyze",
        show_info=True
    )

    if result is None:
        return  # No datasets available

    dataset_name, data = result
    st.success(f"✅ Analyzing: **{dataset_name}**")

    # Continue with your module's analysis code...
```

### Step 3: Remove Old Pattern

**Remove** these old patterns:

```python
# OLD - Don't use this anymore:
if 'current_data' not in st.session_state:
    st.warning("No data loaded...")
    return

data = st.session_state.current_data
```

---

## Complete Examples

### Example 1: PCA Module

```python
# pca.py
from workspace_utils import display_workspace_dataset_selector

def show():
    st.markdown("# 🎯 Principal Component Analysis (PCA)")

    # Dataset selection from workspace
    st.markdown("### 📊 Select Dataset for PCA Analysis")

    result = display_workspace_dataset_selector(
        label="Select dataset from workspace:",
        key="pca_dataset_selector",
        help_text="Choose a dataset to analyze with PCA",
        show_info=True
    )

    if result is None:
        return

    dataset_name, data = result
    st.success(f"✅ Analyzing: **{dataset_name}**")

    # PCA analysis code continues...
```

### Example 2: Transformations Module

```python
# transformations.py
from workspace_utils import display_workspace_dataset_selector

def show():
    st.markdown("# Data Transformations")

    # Dataset selection from workspace
    st.markdown("### 📊 Select Dataset for Transformation")

    result = display_workspace_dataset_selector(
        label="Select dataset from workspace:",
        key="transformations_dataset_selector",
        help_text="Choose a dataset to apply transformations",
        show_info=True
    )

    if result is None:
        return

    selected_dataset_name, data = result
    st.success(f"✅ Working with: **{selected_dataset_name}**")

    # Use selected_dataset_name instead of st.session_state.get('current_dataset')
    # in transformation save operations
```

### Example 3: MLR/DOE Module

```python
# mlr_doe.py
from workspace_utils import display_workspace_dataset_selector

def show():
    st.markdown("# 🧪 Multiple Linear Regression & Design of Experiments")

    # Dataset selection from workspace
    st.markdown("### 📊 Select Dataset for MLR/DOE Analysis")

    result = display_workspace_dataset_selector(
        label="Select dataset from workspace:",
        key="mlr_doe_dataset_selector",
        help_text="Choose a dataset for MLR/DOE analysis",
        show_info=True
    )

    if result is None:
        return

    dataset_name, data = result
    st.success(f"✅ Analyzing: **{dataset_name}**")

    # MLR/DOE analysis code continues...
```

---

## Important Notes

### 1. Unique Keys

**Each module MUST use a unique key** for the selector:

```python
# Good - unique keys
key="pca_dataset_selector"           # PCA module
key="transformations_dataset_selector"  # Transformations module
key="mlr_doe_dataset_selector"       # MLR/DOE module
key="qc_training_data_selector"      # Quality Control - training data
key="qc_test_data_selector"          # Quality Control - test data

# Bad - duplicate keys will cause conflicts!
key="dataset_selector"  # Too generic, don't use the same key twice
```

### 2. Variable Names

The selector returns a tuple:

```python
result = display_workspace_dataset_selector(...)
if result is None:
    return  # No datasets available

dataset_name, data = result  # Unpack the tuple
```

Use meaningful variable names:
- `dataset_name` - String name of the selected dataset
- `data` - pandas DataFrame with the actual data

### 3. Dataset Name Usage

If you need to save transformations or reference the dataset later:

```python
# Good - use the returned dataset_name
base_name = dataset_name.split('.')[0]
transformed_name = f"{base_name}.{transform_code}"

# Bad - don't rely on session state
# base_name = st.session_state.get('current_dataset')  # OLD PATTERN
```

### 4. Optional Parameters

The selector supports optional parameters:

```python
result = display_workspace_dataset_selector(
    label="Custom label text",           # Label for selectbox
    key="unique_key",                     # Unique widget key (REQUIRED)
    help_text="Custom help text",         # Help tooltip
    show_info=True                        # Show dataset metrics (default: True)
)
```

To hide the metrics, set `show_info=False`:

```python
result = display_workspace_dataset_selector(
    label="Select dataset:",
    key="my_selector",
    show_info=False  # Won't show samples/variables metrics
)
```

---

## What the Selector Shows

With `show_info=True`, users see:

```
📊 Select Dataset for Analysis

[Dropdown with all workspace datasets]

✅ Analyzing: NIR_Data.snv

Samples    Variables    Numeric
   100        105         104
```

The selector automatically collects datasets from:
- `st.session_state.current_data` (active dataset)
- `st.session_state.transformation_history` (all transformed datasets)
- `st.session_state.split_datasets` (PCA splits, prefixed with "Split: ")

---

## Benefits

### For Users:
✅ Select any dataset without switching pages
✅ See all available datasets in one place
✅ Quickly switch between datasets for comparison
✅ Access transformed data and splits directly

### For Developers:
✅ Consistent UI across all modules
✅ 3-line implementation
✅ Automatic dataset collection
✅ Built-in error handling

---

## Testing Checklist

When adding to a new module:

- [ ] Import added at top of file
- [ ] Selector added in `show()` function
- [ ] Unique key used (not duplicated)
- [ ] Old `st.session_state.current_data` pattern removed
- [ ] Module works with datasets from workspace
- [ ] Module works with transformed datasets
- [ ] Module works with split datasets
- [ ] No errors when workspace is empty

---

## Future Modules

For **any new analysis module**, just follow the 3-step pattern:

1. Import `display_workspace_dataset_selector`
2. Add selector at top of `show()` function
3. Use returned `dataset_name` and `data`

That's it! Your module now has consistent dataset selection.

---

## Contact

For questions or support:
- **Developer**: ChemometricSolutions
- **File**: `workspace_utils.py`
- **Documentation**: This guide
