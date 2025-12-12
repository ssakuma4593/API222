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
RAW_GTD_PATH = "data/globalterrorismdb_0522dist.xlsx"


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
    
    def visualize_linear_predictions(self, results, figsize=(10, 6), save_path=None):
        """Visualize linear regression predictions vs actual values"""
        if self.linear_model is None:
            print("Error: Linear model not fitted yet. Call fit_linear_regression() first.")
            return
        
        X_test_scaled = self.scaler.transform(results['X_test'])
        y_test = results['y_test']
        y_pred = self.linear_model.predict(X_test_scaled)
        
        # Use percentiles to set better axis limits (focus on 95% of data)
        x_min = np.percentile(y_test, 1)
        x_max = np.percentile(y_test, 99)
        y_min = min(np.percentile(y_pred, 1), x_min)
        y_max = max(np.percentile(y_pred, 99), x_max)
        
        # Add small padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min = max(0, x_min - 0.05 * x_range)
        x_max = x_max + 0.05 * x_range
        y_min = max(0, y_min - 0.05 * y_range)
        y_max = y_max + 0.05 * y_range
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot: predicted vs actual
        axes[0].scatter(y_test, y_pred, alpha=0.5, s=10)
        # Perfect prediction line using the axis limits
        plot_min = min(x_min, y_min)
        plot_max = max(x_max, y_max)
        axes[0].plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_xlabel('Actual Severity')
        axes[0].set_ylabel('Predicted Severity')
        axes[0].set_title(f'Linear Regression: Predicted vs Actual\nTest R² = {results["test_r2"]:.4f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot with better scaling
        residuals = y_test - y_pred
        pred_min = np.percentile(y_pred, 1)
        pred_max = np.percentile(y_pred, 99)
        res_min = np.percentile(residuals, 1)
        res_max = np.percentile(residuals, 99)
        pred_range = pred_max - pred_min
        res_range = res_max - res_min
        
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlim(max(0, pred_min - 0.05 * pred_range), pred_max + 0.05 * pred_range)
        axes[1].set_ylim(res_min - 0.1 * res_range, res_max + 0.1 * res_range)
        axes[1].set_xlabel('Predicted Severity')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Linear Regression: Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Linear regression visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_linear_predictions_log_scale(self, results, figsize=(10, 6), save_path=None):
        """Visualize linear regression predictions vs actual values on log scale"""
        if self.linear_model is None:
            print("Error: Linear model not fitted yet. Call fit_linear_regression() first.")
            return
        
        X_test_scaled = self.scaler.transform(results['X_test'])
        y_test = results['y_test']
        y_pred = self.linear_model.predict(X_test_scaled)
        
        # Filter out zeros and negative values for log scale
        mask = (y_test > 0) & (y_pred > 0)
        y_test_log = y_test[mask]
        y_pred_log = y_pred[mask]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot: predicted vs actual on log scale
        axes[0].scatter(y_test_log, y_pred_log, alpha=0.5, s=10)
        # Perfect prediction line on log scale
        min_val = min(y_test_log.min(), y_pred_log.min())
        max_val = max(y_test_log.max(), y_pred_log.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Actual Severity (log scale)')
        axes[0].set_ylabel('Predicted Severity (log scale)')
        axes[0].set_title(f'Linear Regression: Predicted vs Actual (Log Scale)\nTest R² = {results["test_r2"]:.4f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, which='both')
        
        # Residual plot on log scale (log of predicted vs residuals)
        residuals_log = y_test_log - y_pred_log
        axes[1].scatter(y_pred_log, residuals_log, alpha=0.5, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Predicted Severity (log scale)')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Linear Regression: Residual Plot (Log Scale)')
        axes[1].grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Linear regression log-scale visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_linear_regression_annotated(self, results, figsize=(10, 8), save_path=None):
        """Create an annotated log-scale visualization similar to the reference image"""
        if self.linear_model is None:
            print("Error: Linear model not fitted yet. Call fit_linear_regression() first.")
            return
        
        X_test_scaled = self.scaler.transform(results['X_test'])
        y_test = results['y_test']
        y_pred = self.linear_model.predict(X_test_scaled)
        
        # Filter out zeros and negative values for log scale
        mask = (y_test > 0) & (y_pred > 0)
        y_test_log = y_test[mask]
        y_pred_log = y_pred[mask]
        
        # Calculate R² on the filtered data
        from sklearn.metrics import r2_score
        r2_filtered = r2_score(y_test_log, y_pred_log)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Scatter plot: predicted vs actual on log scale
        # Use red color with transparency to match the reference
        ax.scatter(y_test_log, y_pred_log, alpha=0.4, s=15, color='red', edgecolors='none')
        
        # Perfect prediction line (black dashed)
        min_val = min(y_test_log.min(), y_pred_log.min())
        max_val = max(y_test_log.max(), y_pred_log.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
        
        # Set log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Actual Severity (Log Scale)', fontsize=12)
        ax.set_ylabel('Predicted Severity (Log Scale)', fontsize=12)
        ax.set_title(f'Linear Regression (R² = {r2_filtered:.2f})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
        
        # Add annotation with arrow pointing to the main cluster
        # Find the center of the main cluster (median values)
        median_actual = np.median(y_test_log)
        median_pred = np.median(y_pred_log)
        
        # Calculate the position for the annotation
        # Place it to the upper right of the cluster
        annotation_x = median_actual * 1.5
        annotation_y = median_pred * 0.7
        
        # Text describing the model behavior
        annotation_text = ("Shows systematic underestimation of high-severity events,\n"
                           "with predictions tightly clustered and failing to capture outliers.")
        
        # Add annotation with arrow
        ax.annotate(annotation_text,
                   xy=(median_actual, median_pred),  # Point to annotate
                   xytext=(annotation_x, annotation_y),  # Text position
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black', lw=1.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Annotated linear regression visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_lasso_predictions(self, results, figsize=(10, 6), save_path=None):
        """Visualize LASSO regression predictions vs actual values"""
        if self.lasso_model is None:
            print("Error: LASSO model not fitted yet. Call fit_lasso_regression() first.")
            return
        
        X_test_scaled = self.scaler.transform(results['X_test'])
        y_test = results['y_test']
        y_pred = self.lasso_model.predict(X_test_scaled)
        
        # Use percentiles to set better axis limits (focus on 95% of data)
        x_min = np.percentile(y_test, 1)
        x_max = np.percentile(y_test, 99)
        y_min = min(np.percentile(y_pred, 1), x_min)
        y_max = max(np.percentile(y_pred, 99), x_max)
        
        # Add small padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min = max(0, x_min - 0.05 * x_range)
        x_max = x_max + 0.05 * x_range
        y_min = max(0, y_min - 0.05 * y_range)
        y_max = y_max + 0.05 * y_range
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot: predicted vs actual
        axes[0].scatter(y_test, y_pred, alpha=0.5, s=10)
        # Perfect prediction line using the axis limits
        plot_min = min(x_min, y_min)
        plot_max = max(x_max, y_max)
        axes[0].plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_xlabel('Actual Severity')
        axes[0].set_ylabel('Predicted Severity')
        axes[0].set_title(f'LASSO Regression: Predicted vs Actual\nTest R² = {results["test_r2"]:.4f}, Alpha = {results["best_alpha"]:.6f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot with better scaling
        residuals = y_test - y_pred
        pred_min = np.percentile(y_pred, 1)
        pred_max = np.percentile(y_pred, 99)
        res_min = np.percentile(residuals, 1)
        res_max = np.percentile(residuals, 99)
        pred_range = pred_max - pred_min
        res_range = res_max - res_min
        
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlim(max(0, pred_min - 0.05 * pred_range), pred_max + 0.05 * pred_range)
        axes[1].set_ylim(res_min - 0.1 * res_range, res_max + 0.1 * res_range)
        axes[1].set_xlabel('Predicted Severity')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('LASSO Regression: Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LASSO regression visualization saved to {save_path}")
        
        plt.show()
    
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
    
    def visualize_model_comparison(self, linear_results, lasso_results, figsize=(14, 6), save_path=None):
        """Compare linear and LASSO regression models side by side"""
        if self.linear_model is None or self.lasso_model is None:
            print("Error: Both models must be fitted first.")
            return
        
        # Get predictions
        X_test_linear = self.scaler.transform(linear_results['X_test'])
        X_test_lasso = self.scaler.transform(lasso_results['X_test'])
        y_test = linear_results['y_test']
        y_pred_linear = self.linear_model.predict(X_test_linear)
        y_pred_lasso = self.lasso_model.predict(X_test_lasso)
        
        # Use percentiles to set better axis limits (focus on 95% of data)
        x_min = np.percentile(y_test, 1)
        x_max = np.percentile(y_test, 99)
        y_linear_min = min(np.percentile(y_pred_linear, 1), x_min)
        y_linear_max = max(np.percentile(y_pred_linear, 99), x_max)
        y_lasso_min = min(np.percentile(y_pred_lasso, 1), x_min)
        y_lasso_max = max(np.percentile(y_pred_lasso, 99), x_max)
        
        # Use the same limits for both plots for fair comparison
        plot_min = min(x_min, y_linear_min, y_lasso_min)
        plot_max = max(x_max, y_linear_max, y_lasso_max)
        
        # Add small padding
        plot_range = plot_max - plot_min
        plot_min = max(0, plot_min - 0.05 * plot_range)
        plot_max = plot_max + 0.05 * plot_range
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Linear regression plot
        axes[0].scatter(y_test, y_pred_linear, alpha=0.5, s=10, label='Predictions')
        axes[0].plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlim(plot_min, plot_max)
        axes[0].set_ylim(plot_min, plot_max)
        axes[0].set_xlabel('Actual Severity')
        axes[0].set_ylabel('Predicted Severity')
        axes[0].set_title(f'Linear Regression\nTest R² = {linear_results["test_r2"]:.4f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # LASSO regression plot
        axes[1].scatter(y_test, y_pred_lasso, alpha=0.5, s=10, label='Predictions')
        axes[1].plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label='Perfect Prediction')
        axes[1].set_xlim(plot_min, plot_max)
        axes[1].set_ylim(plot_min, plot_max)
        axes[1].set_xlabel('Actual Severity')
        axes[1].set_ylabel('Predicted Severity')
        axes[1].set_title(f'LASSO Regression\nTest R² = {lasso_results["test_r2"]:.4f}, α = {lasso_results["best_alpha"]:.6f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison visualization saved to {save_path}")
        
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
    
    # Visualize linear regression
    print("\n" + "="*50)
    print("Generating Linear Regression Visualizations")
    print("="*50)
    regressor.visualize_linear_predictions(linear_results, save_path="linear_regression_outputs.png")
    
    # Visualize linear regression on log scale
    regressor.visualize_linear_predictions_log_scale(linear_results, save_path="linear_regression_log_scale.png")
    
    # Visualize linear regression with annotation (matching reference style)
    regressor.visualize_linear_regression_annotated(linear_results, save_path="linear_regression_annotated.png")
    
    # Visualize LASSO regression
    print("\n" + "="*50)
    print("Generating LASSO Regression Visualizations")
    print("="*50)
    regressor.visualize_lasso_predictions(lasso_results, save_path="lasso_regression_outputs.png")
    
    # Visualize LASSO features
    regressor.visualize_lasso_features(top_n=20, save_path="lasso_features.png")
    
    # Compare both models
    print("\n" + "="*50)
    print("Generating Model Comparison Visualization")
    print("="*50)
    regressor.visualize_model_comparison(linear_results, lasso_results, save_path="model_comparison.png")
    
    # Save LASSO features to CSV
    regressor.save_lasso_features("lasso_selected_features.csv")
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    print("="*50)
    print("Generated visualizations:")
    print("  - linear_regression_outputs.png")
    print("  - linear_regression_log_scale.png")
    print("  - linear_regression_annotated.png")
    print("  - lasso_regression_outputs.png")
    print("  - lasso_features.png")
    print("  - model_comparison.png")

