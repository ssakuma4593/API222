"""
Regression Models for Global Terrorism Database
Linear and LASSO regression models for predicting attack severity
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
import matplotlib.pyplot as plt


# Default path to the raw Global Terrorism Database file
RAW_GTD_PATH = "data/globalterrorismdb_2021Jan-June_1222dist.xlsx"


def create_model_ready_from_raw(
    input_path: str = RAW_GTD_PATH,
    output_path: str = "data/gtd_model_ready.csv",
    drop_first: bool = True,
) -> pd.DataFrame:
    """Create a model-ready CSV directly from the raw GTD file.

    This:
    - Loads the original Global Terrorism Database CSV/Excel
    - Cleans casualty fields and constructs ``severity``
    - Keeps ONLY the columns intended for modeling + reference:
        * eventid  (for manual comparison only)
        * severity (target, nkill + nwound)
        * iyear, imonth, iday, latitude, longitude, success, suicide
        * attacktype1_txt, weaptype1_txt, targtype1_txt,
          region_txt, country_txt
    - One-hot encodes the medium-cardinality categoricals
      (attacktype1_txt, weaptype1_txt, targtype1_txt,
       region_txt, country_txt)
    - Saves a compact model-ready CSV while leaving eventid and severity
      available for inspection (they are dropped from X later).
    """
    print(f"Loading raw GTD data from {input_path}...")

    lower = input_path.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(input_path, low_memory=False)
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        df = pd.read_excel(input_path)
    else:
        # Fallback: try CSV then Excel
        try:
            df = pd.read_csv(input_path, low_memory=False)
        except Exception:
            df = pd.read_excel(input_path)

    # Basic casualty cleaning
    for col in ["nkill", "nwound"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Construct severity (and keep it for manual inspection)
    if {"nkill", "nwound"}.issubset(df.columns):
        df["severity"] = df["nkill"] + df["nwound"]
    else:
        df["severity"] = np.nan

    # Restrict to the exact set of columns we care about
    cols_to_keep = [
        "eventid",
        "severity",
        "iyear",
        "imonth",
        "iday",
        "latitude",
        "longitude",
        "success",
        "suicide",
        "attacktype1_txt",
        "weaptype1_txt",
        "targtype1_txt",
        "region_txt",
        "country_txt",
    ]
    existing = [c for c in cols_to_keep if c in df.columns]
    df = df[existing].copy()

    # One-hot encode medium-cardinality categoricals
    medium_cat_cols = [
        "attacktype1_txt",
        "weaptype1_txt",
        "targtype1_txt",
        "region_txt",
        "country_txt",
    ]
    present_cats = [c for c in medium_cat_cols if c in df.columns]
    if present_cats:
        df = pd.get_dummies(df, columns=present_cats, drop_first=drop_first)

    # Keep eventid and severity in the CSV (for manual comparison later).
    if "eventid" not in df.columns:
        print("Warning: 'eventid' not found in columns; it will not be available for manual comparison.")

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved model-ready CSV to: {output_path}")

    return df


class TerrorismRegressionModels:
    """Regression models for predicting terrorism attack severity"""
    
    def __init__(self):
        self.lasso_model = None
        self.linear_model = None
        self.scaler = None
        self.feature_names = None
    
    def prepare_regression_data(self, df):
        """Prepare data for regression analysis.

        This function assumes that heavy-duty cleaning and feature
        selection has already been done upstream by creating
        ``data/gtd_model_ready.csv`` from the raw Global Terrorism
        Database file.

        Its responsibility is primarily to:
        - Ensure there is a numeric ``severity`` target
        - Drop remaining non-feature / leakage columns (including
          ``eventid`` and casualty components)
        - One-hot encode any remaining object (categorical) columns
        - Return X (features) and y (target)
        """
        print("Preparing data for regression analysis...")

        # Work on a copy
        regression_df = df.copy()

        # Ensure target severity exists; if not, construct it from nkill/nwound
        if "severity" not in regression_df.columns:
            if {"nkill", "nwound"}.issubset(regression_df.columns):
                regression_df["severity"] = (
                    regression_df["nkill"].fillna(0) + regression_df["nwound"].fillna(0)
                )
            else:
                raise ValueError(
                    "Input dataframe must contain either 'severity' or both 'nkill' and 'nwound'."
                )

        y = regression_df["severity"]

        # Drop non-feature / leakage-prone columns.
        # This intentionally overlaps with earlier cleaning steps as a safeguard
        # in case an older version of the model-ready CSV is used.
        columns_to_drop = []

        # Always drop target / obvious non-features
        base_drop = [
            "severity",
            "total_casualties",
            "eventid",
            "summary",
            "scite1",
            "scite2",
            "scite3",
            "dbsource",
        ]
        for col in base_drop:
            if col in regression_df.columns:
                columns_to_drop.append(col)

        # Drop all columns that start with 'scite', 'nkill', or 'nwound'
        for col in regression_df.columns:
            if col.startswith("scite") or col.startswith("nkill") or col.startswith("nwound"):
                columns_to_drop.append(col)

        # Drop derived/aggregated casualty-like columns
        outcome_keywords = ["casualty", "death", "fatal"]
        aggregation_terms = ["sum", "total", "count", "mean", "avg", "max", "min"]
        for col in regression_df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in outcome_keywords):
                if col not in columns_to_drop and col not in ["severity"]:
                    if any(a in col_lower for a in aggregation_terms):
                        columns_to_drop.append(col)

        # Drop known high-cardinality categoricals and their one-hot variants
        for col in regression_df.columns:
            col_lower = col.lower()
            if col_lower.startswith("gname") or col_lower.startswith("city"):
                columns_to_drop.append(col)

        # De-duplicate and keep only existing columns
        columns_to_drop = sorted(set(c for c in columns_to_drop if c in regression_df.columns))

        if columns_to_drop:
            print(f"Dropping {len(columns_to_drop)} non-feature / leakage-prone columns: {columns_to_drop}")

        X = regression_df.drop(columns=columns_to_drop)

        # Encode any remaining object dtype columns (e.g., if data came from
        # the visualization cleaner rather than the full model-ready CSV).
        X = pd.get_dummies(X, drop_first=True)

        # Simple missing-value handling
        X = X.fillna(0)

        print(f"Prepared {X.shape[1]} features for regression")
        return X, y

    
    def fit_linear_regression(self, df, test_size=0.2, random_state=42):
        """Fit a linear regression model to predict severity"""
        print("\n" + "="*50)
        print("Fitting Linear Regression Model")
        print("="*50)
        
        # Prepare data
        X, y = self.prepare_regression_data(df)
        self.feature_names = X.columns
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Standardize (scale) features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit linear regression
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train_scaled, y_train)
        
        # Evaluate model performance
        train_r2 = self.linear_model.score(X_train_scaled, y_train)
        test_r2 = self.linear_model.score(X_test_scaled, y_test)
        
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        return {
            'model': self.linear_model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def fit_lasso_regression(self, df, test_size=0.2, random_state=42, cv=5, n_jobs=-1):
        """Fit a LASSO regression model with cross-validation to predict severity"""
        print("\n" + "="*50)
        print("Fitting LASSO Regression Model")
        print("="*50)
        
        # Prepare data
        X, y = self.prepare_regression_data(df)
        self.feature_names = X.columns
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Standardize (scale) features
        # LASSO is sensitive to feature scale, so we normalize all predictors
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit LASSO with cross-validation
        # LassoCV automatically tunes the regularization parameter (alpha)
        self.lasso_model = LassoCV(cv=cv, random_state=random_state, n_jobs=n_jobs)
        self.lasso_model.fit(X_train_scaled, y_train)
        
        # Evaluate model performance
        train_r2 = self.lasso_model.score(X_train_scaled, y_train)
        test_r2 = self.lasso_model.score(X_test_scaled, y_test)
        
        print(f"Best alpha: {self.lasso_model.alpha_:.6f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        # Inspect feature importance
        # LASSO shrinks small coefficients toward zero
        coef = pd.Series(self.lasso_model.coef_, index=X.columns)
        important = coef[coef != 0].sort_values(key=abs, ascending=False)
        
        print(f"\nNumber of features selected: {len(important)} out of {len(X.columns)}")
        print(f"\nTop 10 most important features:")
        for i, (feature, coef_value) in enumerate(important.head(10).items(), 1):
            print(f"  {i}. {feature}: {coef_value:.4f}")
        
        return {
            'model': self.lasso_model,
            'best_alpha': self.lasso_model.alpha_,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'important_features': important,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def visualize_lasso_features(self, top_n=20, figsize=(8, 6), save_path=None):
        """Visualize top features from LASSO model"""
        if self.lasso_model is None:
            print("Error: LASSO model not fitted yet. Call fit_lasso_regression() first.")
            return
        
        # Get feature importance
        coef = pd.Series(self.lasso_model.coef_, index=self.feature_names)
        important = coef[coef != 0].sort_values(key=abs, ascending=False)
        
        if len(important) == 0:
            print("No features selected by LASSO")
            return
        
        # Visualize top features
        plt.figure(figsize=figsize)
        important.head(top_n).sort_values().plot(kind='barh')
        plt.title(f"Top {min(top_n, len(important))} Most Important Features (LASSO)")
        plt.xlabel("Coefficient magnitude")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_lasso_features(self, output_path="lasso_selected_features.csv"):
        """Save LASSO selected features to CSV"""
        if self.lasso_model is None:
            print("Error: LASSO model not fitted yet. Call fit_lasso_regression() first.")
            return
        
        # Get feature importance
        coef = pd.Series(self.lasso_model.coef_, index=self.feature_names)
        important = coef[coef != 0].sort_values(key=abs, ascending=False)
        
        # Save to CSV
        important.to_csv(output_path)
        print(f"LASSO selected features saved to {output_path}")
        return important


if __name__ == "__main__":
    # Example usage
    #
    # Workflow:
    #   1) Check if data/gtd_model_ready.csv exists.
    #   2) If it does NOT exist, create it from the raw GTD file via
    #      create_model_ready_from_raw(), which:
    #        - cleans the data
    #        - drops high-cardinality & leakage-prone columns
    #        - one-hot encodes medium-cardinality categoricals
    #        - KEEPS eventid and severity in the CSV for manual comparison.
    #   3) Load data/gtd_model_ready.csv and pass it into the regression
    #      methods; prepare_regression_data() will then drop eventid and
    #      severity (and other non-features) from the feature matrix X
    #      while still using severity as the target y.

    model_ready_path = "data/gtd_model_ready.csv"
    if not os.path.exists(model_ready_path):
        print(f"{model_ready_path} not found. Creating it from raw GTD data...")
        create_model_ready_from_raw(RAW_GTD_PATH, output_path=model_ready_path)

    print(f"Loading model-ready dataset from {model_ready_path}")
    df = pd.read_csv(model_ready_path)

    # Fit regression models (features will exclude eventid and severity;
    # they remain only in the CSV for inspection)
    regressor = TerrorismRegressionModels()

    # Fit linear regression
    linear_results = regressor.fit_linear_regression(df)

    # Fit LASSO regression
    lasso_results = regressor.fit_lasso_regression(df)
    
    # Visualize LASSO features
    regressor.visualize_lasso_features(top_n=20, save_path="lasso_features.png")
    
    # Save LASSO features to CSV
    regressor.save_lasso_features("lasso_selected_features.csv")

