# Model-Ready Dataset Guide

## Overview

Creating a model-ready CSV dataset that can be used for both linear models (LASSO) and tree-based models (Random Forest, XGBoost) is highly recommended. This guide explains why and how.

## Why Create a Model-Ready CSV?

### Benefits:

1. **Efficiency**: Avoid re-running data cleaning steps every time you train models
2. **Consistency**: Same features used across all model types
3. **Clarity**: Explicit about which columns are used as features
4. **Speed**: Faster model training with pre-cleaned data
5. **Reproducibility**: Easy to share and reproduce results

## Data Cleaning for Different Model Types

### Data Leakage Prevention (Applies to ALL Models)

The same data leakage prevention is needed for both linear and tree-based models:

✅ **Remove target components**: `nkill`, `nwound` (they ARE the target)  
✅ **Remove identifiers**: `eventid` (not predictive)  
✅ **Remove unstructured text**: `summary`, citations (may contain outcomes)  
✅ **Remove derived outcomes**: Aggregated casualty columns  

**Why**: All models would learn spurious patterns from these columns, giving artificially high performance that won't generalize.

### Categorical Variable Handling (Differs by Model Type)

#### Linear Models (LASSO, Linear Regression)
- **Require**: Numeric features only
- **Solution**: One-hot encode categorical variables
- **Challenge**: High-cardinality categoricals (group names, cities) create thousands of features

#### Tree-Based Models (Random Forest, XGBoost)
- **Can handle**: Categorical variables directly (some implementations)
- **OR**: Can use one-hot encoded features (like linear models)
- **Advantage**: Better at handling high-cardinality categoricals naturally
- **Note**: Many sklearn tree models still require numeric input, so encoding may still be needed

## Model-Ready Dataset Structure

The `create_model_ready_dataset()` function creates a CSV with:

### Included Columns:
- ✅ **Target variable**: `severity` (nkill + nwound) - kept for reference
- ✅ **Numeric features**: `iyear`, `imonth`, `iday`, `latitude`, `longitude`, `success`, `suicide`
- ✅ **Medium-cardinality categoricals**: 
  - `attacktype1_txt` (~9 values)
  - `weaptype1_txt` (~13 values)
  - `targtype1_txt` (~22 values)
  - `region_txt` (~12 values)
  - `country_txt` (~150-200 values)

### Excluded Columns:
- ❌ **Data leakage columns**: casualty components, identifiers, text
- ❌ **High-cardinality categoricals** (optional): `gname`, `city` - these create too many features

## Using the Model-Ready Dataset

### For Linear Models (LASSO):

```python
from regression_models import TerrorismRegressionModels
import pandas as pd

# Load model-ready dataset
df = pd.read_csv("data/gtd_model_ready.csv")

# Create regressor
regressor = TerrorismRegressionModels()

# This will:
# 1. Drop leakage columns again (if any remain)
# 2. One-hot encode categoricals (selectively, if configured)
# 3. Prepare features for LASSO
X, y = regressor.prepare_regression_data(df)
```

### For Tree-Based Models:

```python
from regression_models import TerrorismRegressionModels
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load model-ready dataset
df = pd.read_csv("data/gtd_model_ready.csv")

# Option 1: Use one-hot encoded features (like linear models)
regressor = TerrorismRegressionModels()
X, y = regressor.prepare_regression_data(df)  # One-hot encodes all categoricals
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# Option 2: Selective encoding for tree models
# Trees can handle categoricals better, so encode only medium-cardinality ones
# (Implementation would need custom handling)
```

## Creating the Model-Ready Dataset

### Using the Function:

```python
from regression_models import TerrorismRegressionModels
import pandas as pd

# Load your cleaned GTD data
df = pd.read_csv("data/gtd_cleaned.csv")

# Create model-ready dataset
regressor = TerrorismRegressionModels()
model_ready_df = regressor.create_model_ready_dataset(
    df,
    output_path="data/gtd_model_ready.csv",
    drop_high_cardinality=True,  # Drop gname, city, etc.
    keep_severity=True            # Keep target variable in dataset
)
```

### What It Does:

1. **Removes data leakage columns** (same logic as `prepare_regression_data`)
2. **Optionally drops high-cardinality columns** (group names, cities)
3. **Keeps categorical columns as-is** (can encode later based on model type)
4. **Saves to CSV** for easy reuse

### Output:
- CSV file with feature-selected columns
- No data leakage
- Ready for both linear and tree-based models
- Diagnostic output showing what was dropped and why

## Feature Count Comparison

### Before (encoding all categoricals):
- `gname`: ~3,500 groups → **3,500 features**
- `city`: ~1,200 cities → **1,200 features**
- `country_txt`: ~180 countries → **179 features**
- Other categoricals: ~50 features
- **Total: ~6,000 features** ❌

### After (model-ready dataset, selective encoding):
- `country_txt`: ~180 countries → **179 features** (if encoded)
- `attacktype1_txt`: ~9 types → **8 features**
- `weaptype1_txt`: ~13 types → **12 features**
- `targtype1_txt`: ~22 types → **21 features**
- `region_txt`: ~12 regions → **11 features**
- Numeric features: ~10 features
- **Total: ~250 features** ✅

## Recommendations

### 1. **Always Use Model-Ready Dataset**
   - Create once, use many times
   - Ensures consistency across experiments
   - Faster iteration during model development

### 2. **Drop High-Cardinality Columns for Linear Models**
   - Group names (`gname`) and cities (`city`) create too many features
   - Trees can handle them better, but still benefit from limiting features

### 3. **Selective Encoding Strategy**
   - For linear models: encode medium-cardinality categoricals only
   - For tree models: can encode fewer categoricals (trees handle them naturally)

### 4. **Keep Dataset as CSV**
   - Easy to inspect in Excel/spreadsheet tools
   - Version control friendly (if not too large)
   - Shareable across team members

## Example Workflow

```python
# Step 1: Create model-ready dataset (do this once)
from regression_models import TerrorismRegressionModels
import pandas as pd

df_cleaned = pd.read_csv("data/gtd_cleaned.csv")
regressor = TerrorismRegressionModels()
df_model_ready = regressor.create_model_ready_dataset(
    df_cleaned,
    output_path="data/gtd_model_ready.csv",
    drop_high_cardinality=True
)

# Step 2: Use for LASSO (linear model)
df = pd.read_csv("data/gtd_model_ready.csv")
X, y = regressor.prepare_regression_data(df)
lasso_results = regressor.fit_lasso_regression(df)

# Step 3: Use for Random Forest (tree model) - same dataset!
from sklearn.ensemble import RandomForestRegressor
X, y = regressor.prepare_regression_data(df)  # Or handle encoding differently
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# Step 4: Compare models using same features
models_dict = {
    'LASSO': regressor.lasso_model,
    'Random Forest': rf_model
}
comparison = regressor.compare_models(
    models_dict, X_train, X_test, y_train, y_test,
    scalers_dict={'LASSO': regressor.scaler}
)
```

## Summary

✅ **Yes, the data cleaning is helpful for tree-based models** - same leakage prevention needed  
✅ **Yes, create a model-ready CSV** - more efficient and consistent  

The model-ready dataset provides:
- Cleaned features (no leakage)
- Reasonable feature count (~200-500 instead of ~6,000)
- Ready for both linear and tree-based models
- Faster model development and iteration

