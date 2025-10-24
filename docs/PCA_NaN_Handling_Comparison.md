# PCA NaN Handling: R-CAT vs Python Implementation

**Document Purpose**: Compare NIPALS missing value handling between R pcaMethods and Python implementation to identify why explained variance percentages differ with NaN data.

---

## 1. R-CAT (pcaMethods Package) Behavior

### Detection
```r
# PCA_model_PCA.r, lines 53-56
nNA <- sum(is.na(M_))
if(nNA > 0) {
  mess <- paste(as.character(nNA), 'missing data present - This is a WARNING message')
  tk_messageBox(message=mess)
}
```

### Processing
- Uses `pcaMethods::pca()` with `method="nipals"`
- NIPALS algorithm handles NaN internally
- **Key**: pcaMethods package documentation needed for exact algorithm

### Variance Calculation (Critical)
```r
# Lines 61-62
sgt <- as.integer(ans[[4]])              # n_components (default)
if(!ans[[6]]) sgt <- sum(apply(M_, 2, var))  # If not scaled: sum(var) on ORIGINAL data
```

**IMPORTANT**:
- `sgt` calculated on **original data M_** before `prep()` preprocessing
- Then data is centered/scaled: `md <- prep(M_, scale=ccs, center=ans[[5]], ...)`
- PCA run on `md$data` (preprocessed)
- But `sgt` still references original variance

### R `var()` with NaN
In R, `var(x)` where `x` contains NA:
```r
var(c(1, 2, NA, 4))  # Returns NA by default
var(c(1, 2, NA, 4), na.rm=TRUE)  # Returns variance of available values
```

**R-CAT uses**: `sum(apply(M_, 2, var))`
- If M_ has NaN: result could be NA unless pcaMethods handles it internally
- Need to verify if `prep()` function removes/imputes NaN before variance calculation

### Diagnostics Restrictions
```r
# PCA_diagnostic_plot_t2vsq.r, lines 11-12
if(sum(PCA$res@missing) > 0) {
  mess <- paste('Not possible to compute Q diagnostics with', sum(PCA$res@missing), 'missing data')
  tk_messageBox(message=mess)
}
```

**Q residuals are disabled with missing data** - reconstruction would be inaccurate.

---

## 2. Python Implementation (Current)

### Detection
```python
# pca_calculations.py, lines 91-98
n_missing_original = np.isnan(X_array).sum()
n_total = X_array.size
pct_missing = (n_missing_original / n_total) * 100
print(f"Missing values: {n_missing_original} / {n_total} ({pct_missing:.2f}%)")
```

### Preprocessing with NaN
```python
# Lines 100-111
X_processed = X_array.copy()
means = np.nanmean(X_processed, axis=0)  # Ignores NaN in mean calculation

if center:
    X_processed = X_processed - means  # NaN remains NaN after subtraction

if scale:
    stds = np.nanstd(X_processed, axis=0)  # Ignores NaN in std calculation
    X_processed = X_processed / stds  # NaN remains NaN after division
```

### NIPALS Algorithm with NaN
```python
# Lines 140-160
# Compute loadings: p = X'*t / (t'*t)
for var_idx in range(n_features):
    var_col = X_work[:, var_idx]
    valid_mask = ~np.isnan(var_col)  # Exclude NaN values
    if np.sum(valid_mask) > 0:
        p[var_idx] = np.dot(var_col[valid_mask], t[valid_mask]) / np.dot(t[valid_mask], t[valid_mask])

# Compute scores: t = X*p
for sample_idx in range(n_samples):
    sample_row = X_work[sample_idx, :]
    valid_mask = ~np.isnan(sample_row)  # Exclude NaN values
    if np.sum(valid_mask) > 0:
        t[sample_idx] = np.dot(sample_row[valid_mask], p[valid_mask])
```

**Method**: Pairwise deletion - only uses available values for each calculation.

### Variance Calculation (CRITICAL DIFFERENCE)
```python
# Lines 174, 201
eigenvalue = np.dot(t, t) / (n_samples - 1)  # Variance of scores
total_variance = np.sum(np.nanvar(X_processed, axis=0, ddof=1))
```

**Key Differences from R**:
1. `total_variance` calculated on **PREPROCESSED** data (after centering/scaling)
2. Uses `np.nanvar()` which excludes NaN from each column's variance
3. `ddof=1` for unbiased estimate (divides by n-1)

### `np.nanvar()` Behavior
```python
import numpy as np
x = np.array([1, 2, np.nan, 4])
np.var(x)       # Returns nan
np.nanvar(x)    # Returns 2.333... (variance of [1, 2, 4])
np.nanvar(x, ddof=1)  # Returns 2.333... with Bessel's correction
```

---

## 3. IDENTIFIED DISCREPANCIES

### A. Total Variance Reference Point

| Aspect | R-CAT | Python |
|--------|-------|--------|
| **Data used** | Original `M_` | Preprocessed `X_processed` |
| **Timing** | Before `prep()` | After centering/scaling |
| **Impact** | Variance in original scale | Variance in preprocessed scale |

**For UNSCALED data**:
- Centering doesn't change variance
- Should be equivalent IF NaN handling is the same

**For SCALED data**:
- R: Uses original variance sum
- Python: Uses scaled variance (each var ≈ 1.0, sum ≈ n_features)
- **This could explain discrepancies!**

### B. NaN Variance Calculation

**R**: Unknown - depends on `pcaMethods::prep()` and how it handles NA
- May remove entire columns with too many NA
- May impute NA before variance calculation
- May use `na.rm=TRUE` equivalent

**Python**: `np.nanvar(X_processed, axis=0, ddof=1)`
- Calculates variance per column using only available values
- Each column may have different n_available
- No missing data imputation

**Hypothesis**: Different denominators in variance calculation
- R might use n_total for all columns
- Python uses n_available per column via `np.nanvar()`

### C. Eigenvalue Calculation

Both implementations appear to use:
```
eigenvalue = t' * t / (n - 1)
```

Where `t` is the score vector. This should be identical IF the NIPALS iterations converge to the same solution.

**Potential difference**: If NaN handling differs in loadings/scores computation, the score vectors `t` will differ, leading to different eigenvalues.

---

## 4. TESTING STRATEGY

### Test Cases to Compare

1. **No Missing Data**
   - Python and R-CAT should match exactly
   - If they differ: algorithm implementation error

2. **Random Missing Data (5-10%)**
   - Check variance explained percentages
   - Compare eigenvalues directly
   - Compare score vectors

3. **Column with Many Missing (>50%)**
   - Check if R drops column vs Python keeps it
   - Compare total variance calculation

4. **Scaled vs Unscaled**
   - Compare with scale=FALSE and scale=TRUE
   - Verify variance calculation reference point

### Debug Output Comparison

**Python provides** (already implemented):
```
Missing values: X / Y (Z%)
Total variance (sum of column variances): A.BBBBBB
Eigenvalues: [...]
Explained variance ratio: [...]
PC1: XX.XX%
```

**Need from R-CAT**:
- Total variance (`PCA$sgt`)
- Eigenvalues (`PCA$res@sDev^2`)
- Variance explained (`PCA$res@R2`)
- Number of missing values (`sum(PCA$res@missing)`)

---

## 5. RECOMMENDATIONS

### Immediate Actions

1. **Test with clean data (no NaN)**
   - Verify Python matches R-CAT exactly
   - If not matching: fix algorithm bug first

2. **Add variance calculation option**
   ```python
   def compute_pca_nipals(
       X,
       n_components,
       center=True,
       scale=False,
       variance_on_original=False,  # NEW OPTION
       ...
   ):
       # If variance_on_original=True:
       #   Calculate total_variance on X_array BEFORE preprocessing
       # Else:
       #   Calculate total_variance on X_processed AFTER preprocessing (current)
   ```

3. **Document NaN handling differences**
   - Add warning message if NaN detected and results may differ from R-CAT
   - Suggest testing without NaN first

### Long-term Solutions

**Option A: Match R-CAT exactly**
- Research pcaMethods source code
- Implement identical NaN handling
- Pros: Identical results
- Cons: May require R package dependency analysis

**Option B: Provide alternative methods**
- Add parameter: `nan_handling='pairwise' | 'listwise' | 'impute'`
- Document differences
- Allow user to choose
- Pros: Flexibility
- Cons: More complex API

**Option C: Remove NaN support**
- Require complete data
- Match sklearn PCA behavior
- Pros: Simple, predictable
- Cons: Users must preprocess data

---

## 6. NEXT STEPS (DEFERRED IMPLEMENTATION)

1. ✅ **Document differences** (this file)
2. ⏸️ Test Python vs R-CAT with clean data (no NaN)
3. ⏸️ Test Python vs R-CAT with 10% random NaN
4. ⏸️ Analyze pcaMethods source code if discrepancies persist
5. ⏸️ Implement fix based on findings

**Current Status**: Documented for future investigation. Implementation awaits test results comparing Python vs R-CAT output on specific datasets.

---

## APPENDIX: R pcaMethods Package Research

### Installation Check
```r
library(pcaMethods)
?pca
?prep
```

### Key Functions to Investigate

1. **`prep()`** - Data preprocessing
   - How are NA values handled?
   - Does it impute, remove, or skip?

2. **`pca(..., method="nipals")`** - NIPALS implementation
   - Source code location
   - NA handling strategy

3. **`@sDev`, `@R2`, `@R2cum`** - Variance outputs
   - How is total variance calculated internally?
   - Does it use original or preprocessed data variance?

### Command to Extract R Source
```r
# In R console
library(pcaMethods)
methods(pca)
getMethod("pca", "matrix")
```

**TODO**: Execute these commands and compare with Python implementation.

---

**Last Updated**: 2025-10-24
**Status**: Documentation complete, implementation deferred pending test results
