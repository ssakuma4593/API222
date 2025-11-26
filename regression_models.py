"""
Regression Models for Global Terrorism Database
Linear and LASSO regression models for predicting attack severity
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
import matplotlib.pyplot as plt


class TerrorismRegressionModels:
    """Regression models for predicting terrorism attack severity"""
    
    def __init__(self):
        self.lasso_model = None
        self.linear_model = None
        self.scaler = None
        self.feature_names = None
    
    def prepare_regression_data(self, df):
        """Prepare data for regression analysis.

        Expects a \"model-ready\" dataframe where obvious data leakage
        columns (identifiers, text, casualty components) have already
        been removed. If you start from the raw cleaned GTD data,
        prefer calling create_model_ready_dataset() once and then
        re-using the saved CSV.
        """
        print("Preparing data for regression analysis...")
        
        # Create a copy to avoid modifying original data
        regression_df = df.copy()
        
        # Define target (y) and features (X)
        # Target: total casualties = nkill + nwound
        regression_df['severity'] = regression_df['nkill'].fillna(0) + regression_df['nwound'].fillna(0)
        y = regression_df['severity']
        
        # Drop target, outcome variables, identifiers, and unstructured text to prevent data leakage
        columns_to_drop = [
            'severity',           # Target variable
            'total_casualties',   # Same as target (nkill + nwound)
            'eventid',            # Identifier, not a feature
            'summary',            # Unstructured text, not useful as a feature
        ]
        
        # Drop citation columns (scite1, scite2, scite3) - unstructured text
        for col in regression_df.columns:
            if col.startswith('scite'):
                columns_to_drop.append(col)
        
        # Drop all columns that start with 'nkill' or 'nwound' (all casualty-related)
        # These include: nkill, nwound, nkillter, nkillus, nwoundte, nwoundus, etc.
        for col in regression_df.columns:
            if col.startswith('nkill') or col.startswith('nwound'):
                columns_to_drop.append(col)
        
        # Find and drop any other columns that might be outcomes
        # Look for columns with names suggesting they're derived from casualties
        outcome_keywords = ['casualty', 'death', 'fatal']
        for col in regression_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in outcome_keywords):
                if col not in columns_to_drop:
                    # Check if it's an aggregated/derived column
                    if any(agg in col_lower for agg in ['sum', 'total', 'count', 'mean', 'avg', 'max', 'min']):
                        columns_to_drop.append(col)
        
        # Remove duplicates and only drop columns that exist
        columns_to_drop = list(set([col for col in columns_to_drop if col in regression_df.columns]))
        
        if columns_to_drop:
            print(f"Dropping {len(columns_to_drop)} columns to prevent data leakage: {columns_to_drop}")
        
        X = regression_df.drop(columns=columns_to_drop)
        
        # Encode categorical variables
        # Turn all non-numeric columns into dummy/indicator variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Handle missing values
        # Fill NaN with 0 (or could use mean/median imputation)
        X = X.fillna(0)
        
        print(f"Prepared {X.shape[1]} features for regression")
        return X, y

    def create_model_ready_dataset(
        self,
        df: pd.DataFrame,
        output_path: str = "data/gtd_model_ready.csv",
        drop_high_cardinality: bool = True,
    ) -> pd.DataFrame:
        """Create and save a \"model‑ready\" dataset for both linear and tree models.

        This:
        - Constructs the severity target (nkill + nwound)
        - Drops data‑leakage columns (casualty components, identifiers, text, derived outcomes)
        - Optionally drops very high‑cardinality categoricals (e.g., group name, city)
        - Keeps medium‑cardinality categoricals (attack type, weapon type, target type,
          region, country) plus numeric features

        The resulting CSV lives under data/ so it is ignored by git.
        """
        print("\n" + "=" * 60)
        print("Creating model‑ready dataset (for linear and tree models)")
        print("=" * 60)

        model_df = df.copy()

        # Ensure severity exists
        if "severity" not in model_df.columns:
            model_df["severity"] = model_df["nkill"].fillna(0) + model_df["nwound"].fillna(0)

        columns_to_drop = []

        # 1) Identifiers
        if "eventid" in model_df.columns:
            columns_to_drop.append("eventid")

        # 2) Unstructured text and citations
        if "summary" in model_df.columns:
            columns_to_drop.append("summary")
        for col in model_df.columns:
            if col.startswith("scite"):
                columns_to_drop.append(col)

        # 3) Casualty components (data leakage)
        for col in model_df.columns:
            if col.startswith("nkill") or col.startswith("nwound"):
                columns_to_drop.append(col)

        # 4) Derived/aggregated outcome columns (death/casualty/fatal + agg words)
        outcome_keywords = ["casualty", "death", "fatal"]
        aggregation_terms = ["sum", "total", "count", "mean", "avg", "max", "min"]
        for col in model_df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in outcome_keywords):
                if col not in columns_to_drop and col not in ["severity"]:
                    if any(a in col_lower for a in aggregation_terms):
                        columns_to_drop.append(col)

        # 5) Optional: drop very high‑cardinality categoricals that explode feature count
        if drop_high_cardinality:
            for candidate in ["gname", "city", "provstate", "location"]:
                if candidate in model_df.columns:
                    n_unique = model_df[candidate].nunique()
                    if n_unique > 100:
                        print(f"  Dropping high‑cardinality column '{candidate}' ({n_unique} unique values)")
                        columns_to_drop.append(candidate)

        # Always drop duplicate of severity if present
        if "total_casualties" in model_df.columns:
            columns_to_drop.append("total_casualties")

        # De‑duplicate and filter to existing columns
        columns_to_drop = sorted(set(c for c in columns_to_drop if c in model_df.columns))

        if columns_to_drop:
            print(f"\nDropping {len(columns_to_drop)} columns from model‑ready dataset:")
            print(f"  {columns_to_drop}")

        model_df = model_df.drop(columns=columns_to_drop)

        # Simple summary
        n_rows, n_cols = model_df.shape
        cat_cols = model_df.select_dtypes(include=["object"]).columns.tolist()
        num_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()

        print(f"\nFinal model‑ready shape: {n_rows} rows × {n_cols} columns")
        print(f"  Numeric columns:     {len(num_cols)}")
        print(f"  Categorical columns: {len(cat_cols)}")

        # Save under data/ (ignored by git as per README)
        model_df.to_csv(output_path, index=False)
        print(f"\nSaved model‑ready CSV to: {output_path}")

        return model_df
    
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
    import pandas as pd
    from data_processor import TerrorismDataProcessor
    
    # Load cleaned data
    df = pd.read_csv("data/gtd_cleaned.csv")
    
    # Process data
    processor = TerrorismDataProcessor()
    cleaned_data = processor.clean_data(df)
    
    # Fit regression models
    regressor = TerrorismRegressionModels()
    
    # Fit linear regression
    linear_results = regressor.fit_linear_regression(cleaned_data)
    
    # Fit LASSO regression
    lasso_results = regressor.fit_lasso_regression(cleaned_data)
    
    # Visualize LASSO features
    regressor.visualize_lasso_features(top_n=20, save_path="lasso_features.png")
    
    # Save LASSO features to CSV
    regressor.save_lasso_features("lasso_selected_features.csv")

