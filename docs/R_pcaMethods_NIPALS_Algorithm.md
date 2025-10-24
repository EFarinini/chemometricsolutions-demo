# R pcaMethods NIPALS Algorithm - Exact Specification

**Source**: pcaMethods Bioconductor package, Wold (1966)

---

## Critical Findings

### **1. Total Variance Reference**
```r
# PCA_model_PCA.r lines 61-62
sgt <- as.integer(ans[[4]])
if(!ans[[6]]) sgt <- sum(apply(M_, 2, var))
```

**KEY**: Total variance calculated on **ORIGINAL data M_** BEFORE `prep()` preprocessing!
- If unscaled: `sgt = sum(var(M_))` where `var()` uses denominator (n-1)
- If scaled: `sgt = n_components` (this looks like a bug or placeholder)

### **2. NIPALS Algorithm Parameters**

**From R documentation**:
- **Max iterations**: 5000 (not 500!)
- **Convergence criterion**: `(T_old - T)^T(T_old - T) > threshold`
  - Note: Uses `>` not `<` (stops when change is SMALL)
  - Mathematically: `sum((t_old - t)^2) < threshold`
- **Threshold**: 1e-06
- **Preprocessing**: Requires pre-centered, pre-scaled matrix

### **3. NIPALS Iteration Steps**

```
INITIALIZATION:
1. X_h = X (preprocessed data)
2. h = 1 (component index)

FOR EACH COMPONENT h:
  3. t_h = first column of X_h (or any column without all NaN)

  ITERATE UNTIL CONVERGENCE:
    4. p_h = X_h' t_h / (t_h' t_h)
       - For missing values: skip NaN elements when computing dot products

    5. Normalize: p_h = p_h / sqrt(p_h' p_h)

    6. t_h = X_h p_h / (p_h' p_h)
       - Since p_h is unit length: t_h = X_h p_h
       - For missing values: skip NaN elements when computing dot products

    7. Check convergence: sum((t_h_old - t_h)^2) < threshold

  8. Eigenvalue: λ_h = t_h' t_h  (NO division by n-1!)

  9. Deflation: X_(h+1) = X_h - t_h p_h'

  10. h = h + 1
```

### **4. Variance Calculation**

**Eigenvalues**:
```
λ_h = t_h' t_h
```
NO division by (n-1)! This is the sum of squared scores.

**Total Variance** (for unscaled data):
```r
sgt = sum(apply(M_, 2, var))
    = sum(colSums(M_^2) / (n-1))  # R's var() divides by n-1
    = sum(SS_per_column) / (n-1)
```
Where M_ is ORIGINAL data (before centering/scaling).

**Variance Explained Ratio**:
```
R^2_h = λ_h / sgt
```

But wait - this is inconsistent! λ_h doesn't divide by (n-1), but sgt does...

Let me recalculate. If:
- `sgt = sum(SS_original) / (n-1)` (sum of variances of original data)
- `λ_h = t_h' t_h` (sum of squared scores, no division)

Then for centered data where sum of SS equals trace of covariance matrix times (n-1):
- `trace(Cov) = sum(var(columns)) = sgt`
- `trace(Cov) * (n-1) = sum(SS_centered)`

Hmm, this is getting confusing. Let me think differently.

**Alternative interpretation**:
If data is centered but not scaled:
- Original variance: `sgt = sum(var(original columns))`
- After centering, the total SS is: `sum(SS_centered) = sgt * (n-1)`
- NIPALS extracts λ_h = t' t from centered data
- Variance ratio: `λ_h / [sgt * (n-1)]` or `λ_h / sgt`?

Looking at R output: `@R2` is variance explained ratio, `@sDev` is standard deviation.
- `@sDev^2 = eigenvalue = λ`
- `@R2 = variance explained ratio`

If R uses: `R^2 = λ / sgt`, then we need to understand what `sgt` represents.

### **5. Missing Value Handling**

**During Preprocessing** (prep function):
- NaN values are LEFT AS NaN
- Mean calculation: uses available values (na.rm=TRUE equivalent)
- Centering: `X - mean` keeps NaN as NaN
- Scaling: `(X - mean) / sd` keeps NaN as NaN

**During NIPALS Iterations**:
- Pairwise deletion: skip NaN in dot products
- "Interpolates missing point using least squares fit but gives missing data no influence"
- This means: NaN doesn't affect p or t calculation, but after iteration, the NaN position could theoretically be filled with t * p (though it's not used in next iteration)

**Key Quote**: "Successive iterations refine the missing value by simply multiplying the score and the loading for that point."

This suggests iterative imputation during convergence!

---

## Algorithm Comparison: R vs Current Python

| Aspect | R pcaMethods | Current Python | Match? |
|--------|-------------|----------------|---------|
| Max iterations | 5000 | 500 | ❌ |
| Convergence | sum((t_old-t)^2) < 1e-6 | ||t_old-t|| < 1e-6 | ✓ (equivalent) |
| Eigenvalue | t' t | t' t / (n-1) | ❌ |
| Total variance | sum(var(original)) | sum(var(processed)) | ❌ |
| Total var timing | Before preprocessing | After preprocessing | ❌ |
| NaN in preprocessing | Keeps as NaN | Keeps as NaN | ✓ |
| NaN in NIPALS | Pairwise deletion | Pairwise deletion | ✓ |

---

## Implementation Plan

### Fix 1: Total Variance on Original Data
```python
# Calculate BEFORE preprocessing
total_variance_original = np.sum(np.nanvar(X_array, axis=0, ddof=1))

# Then do preprocessing
X_processed = preprocess(X_array, center, scale)

# Use original total variance for variance ratios
```

### Fix 2: Eigenvalue Calculation
```python
# Don't divide by (n-1)
eigenvalue = np.dot(t, t)  # Not / (n_samples - 1)
```

### Fix 3: Adjust Variance Ratio
If eigenvalue is `t' t` and total variance is `sum(var(original))`:
```python
explained_variance_ratio = eigenvalue / (total_variance_original * (n_samples - 1))
```

Wait, this needs careful thought. Let me work out the math:

**Original data** (before centering):
- Variance of column j: `var_j = SS_original_j / (n-1)`
- Total variance: `total_var = sum(var_j) = sum(SS_original_j) / (n-1)`

**Centered data**:
- After centering: `X_centered = X - mean(X)`
- SS of centered data: `SS_centered_j ≈ SS_original_j` (approximately equal for centered data)
- Total SS: `sum(SS_centered_j) ≈ sum(SS_original_j) = total_var * (n-1)`

**NIPALS on centered data**:
- Extracts eigenvalue: `λ = t' t`
- This λ comes from the centered data space
- Variance explained: `λ / [total_var * (n-1)]`?

But R code shows: `@R2 = λ / sgt` where `sgt = total_var`

So: `R^2 = (t' t) / sum(var(original))`

This means R is comparing:
- Numerator: sum of squared scores (no normalization)
- Denominator: sum of variances (normalized by n-1)

This is inconsistent units! Unless...

**Alternative**: R's pcaMethods internally adjusts this. The @R2 might be calculated differently than sgt suggests.

---

## Next Steps

1. Implement exact algorithm with all fixes
2. Test on Two Whiskeys (0 NaN) - should still match ✓
3. Test on Elaboration 4 (48 NaN) - should now match R-CAT exactly
4. If still doesn't match: investigate prep() function in R

---

**Status**: Algorithm documented, ready for implementation
**Target**: Python PC1 = 23.40% (matching R-CAT)
**Current**: Python PC1 = 23.12% (0.28% error)
