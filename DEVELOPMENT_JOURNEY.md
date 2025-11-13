# Development Journey: Regression Models and Data Leakage Fixes

## Initial Problem: Perfect R² Scores (R² = 1.0)

When we first implemented the regression models, we encountered suspiciously perfect results:
- **Linear Regression**: Train R² = 1.0000, Test R² = 1.0000
- **LASSO Regression**: Train R² = 1.0000, Test R² = 1.0000
- **Features selected**: Only 1 feature (`total_casualties`)

This was a clear red flag indicating **data leakage** - the model was using information that directly relates to the target variable.

## Step 1: Identifying Data Leakage - `total_casualties`

**Problem Found**: The `total_casualties` column was included in the features, but it's calculated as `nkill + nwound`, which is exactly the same as our target variable `severity`.

**Solution**: 
- Excluded `total_casualties` from features
- Also excluded `nkill` and `nwound` (direct components of target)
- Excluded `eventid` (identifier, not a predictive feature)

**Result**: 
- Linear Regression: Train R² = 0.9999, Test R² = 0.1579
- LASSO Regression: Train R² = 0.8738, Test R² = 0.3072
- Features selected: 636 out of 17,072

Better, but still some issues with feature selection.

## Step 2: Removing Unstructured Text Columns

**Problem Found**: The top features included:
- `summary_*` columns (unstructured text descriptions)
- `scite1_*`, `scite2_*`, `scite3_*` columns (citation text)

These unstructured text columns were being one-hot encoded, creating thousands of features based on unique text strings. This was:
1. Not useful for prediction (unstructured text)
2. Creating noise in the model
3. Potentially causing overfitting

**Solution**:
- Excluded `summary` column
- Excluded all `scite*` columns (scite1, scite2, scite3)

**Result**: 
- Linear Regression: Train R² = 0.9999, Test R² = 0.1579
- LASSO Regression: Train R² = 0.8738, Test R² = 0.3072
- Features selected: 636 out of 17,072

## Step 3: Comprehensive Data Leakage Fix

**Problem Found**: Additional casualty-related columns were still present:
- `nkillter` (number of terrorists killed)
- `nkillus` (number of US citizens killed)
- `nwoundte` (number of terrorists wounded)
- `nwoundus` (number of US citizens wounded)

These are all subsets or components of the total casualties, so they still represent data leakage.

**Solution**:
- Excluded ALL columns starting with `nkill*` or `nwound*`
- This catches all variations: nkill, nwound, nkillter, nkillus, nwoundte, nwoundus, etc.

**Result**:
- Linear Regression: Train R² = 0.9118, Test R² = -30.9203
- LASSO Regression: Train R² = 0.5696, Test R² = 0.0531
- Features selected: 141 out of 6,941

Much more realistic results! The negative test R² for linear regression indicates overfitting, but LASSO shows reasonable performance.

## Step 4: Testing with US-Only Data

**Hypothesis**: Perhaps focusing on a single country would improve model performance.

**Result**: 
- Only **26 records** for US data
- This was insufficient for meaningful regression analysis
- Model performance was poor due to small sample size

**Decision**: Keep all countries in the dataset for better statistical power.

## Final Solution: All Countries with Proper Data Leakage Prevention

**Final Configuration**:
- Dataset: 4,925 records (all countries)
- Excluded columns (12 total):
  - `severity` (target variable)
  - `total_casualties` (same as target)
  - `eventid` (identifier)
  - `summary` (unstructured text)
  - `scite1`, `scite2`, `scite3` (citation text)
  - All `nkill*` and `nwound*` columns (6 variations)

**Final Results**:
- **Linear Regression**: 
  - Train R² = 0.9118
  - Test R² = -30.9203 (indicates overfitting)
  
- **LASSO Regression**:
  - Best alpha = 0.306767
  - Train R² = 0.5696
  - Test R² = 0.0531
  - Features selected: 141 out of 6,941

**Top Features Identified**:
1. `gname_Mouhadine` (terrorist group)
2. `target1_Guesthouse` (target type)
3. City locations (Daletti, Tekalgudem, Tchombangou)
4. `suicide` flag
5. `gname_Taliban` (terrorist group)
6. Weapon subtype (Vehicle)
7. Target types (Open-Air Market, Civilians)

## Key Learnings

1. **Data Leakage Detection**: Perfect R² scores are a red flag - always investigate!
2. **Comprehensive Exclusion**: Need to exclude all variations of outcome-related columns
3. **Unstructured Text**: Text columns should be excluded or properly processed (NLP) before use
4. **Sample Size Matters**: 26 records is insufficient for regression analysis
5. **LASSO Regularization**: Helps prevent overfitting and provides feature selection

## Files Created/Modified

- `regression_models.py` - New module for regression analysis
- `data_processor.py` - Cleaned to remove regression code
- `regression_output.txt` - Full regression results
- `lasso_features.png` - Visualization of top features
- `lasso_selected_features.csv` - CSV of selected features

