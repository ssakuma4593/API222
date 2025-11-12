"""
Data Preprocessing for Global Terrorism Database
Cleans and prepares data for geographic visualization
"""

import pandas as pd
import numpy as np

class TerrorismDataProcessor:
    def __init__(self):
        self.processed_data = None
    
    def clean_data(self, df):
        """Clean and prepare the terrorism data for visualization"""
        print("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # Basic data cleaning
        print("Performing basic data cleaning...")
        
        # Remove records without valid coordinates
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=['latitude', 'longitude'])
        coord_removed = initial_count - len(cleaned_df)
        print(f"Removed {coord_removed} records without valid coordinates")
        
        # Filter out invalid coordinates (outside valid lat/lng ranges)
        cleaned_df = cleaned_df[
            (cleaned_df['latitude'] >= -90) & (cleaned_df['latitude'] <= 90) &
            (cleaned_df['longitude'] >= -180) & (cleaned_df['longitude'] <= 180)
        ]
        invalid_coord_removed = len(df) - coord_removed - len(cleaned_df)
        if invalid_coord_removed > 0:
            print(f"Removed {invalid_coord_removed} records with invalid coordinate ranges")
        
        # Handle missing death counts
        cleaned_df['nkill'] = cleaned_df['nkill'].fillna(0)
        cleaned_df['nwound'] = cleaned_df['nwound'].fillna(0)
        
        # Create total casualties column
        cleaned_df['total_casualties'] = cleaned_df['nkill'] + cleaned_df['nwound']
        
        # Clean text fields
        text_columns = ['country_txt', 'region_txt', 'city', 'gname']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
        
        # Create decade column for temporal analysis
        cleaned_df['decade'] = (cleaned_df['iyear'] // 10) * 10
        
        print(f"Final dataset: {len(cleaned_df):,} records")
        
        self.processed_data = cleaned_df
        return cleaned_df
    
    def create_aggregated_data(self, df, group_by='city'):
        """Create aggregated data for better visualization"""
        print(f"Creating aggregated data grouped by {group_by}...")
        
        # Define aggregation columns
        agg_dict = {
            'eventid': 'count',  # Number of incidents
            'nkill': 'sum',      # Total deaths
            'nwound': 'sum',     # Total wounded
            'total_casualties': 'sum',  # Total casualties
            'latitude': 'mean',  # Average coordinates
            'longitude': 'mean',
            'iyear': ['min', 'max'],  # Year range
            'country_txt': 'first',
            'region_txt': 'first'
        }
        
        # Group by the specified column(s)
        if group_by == 'city':
            group_cols = ['city', 'country_txt']
        elif group_by == 'country':
            group_cols = ['country_txt']
        else:
            group_cols = [group_by]
        
        aggregated = df.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Flatten column names
        aggregated.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col 
            for col in aggregated.columns
        ]
        
        # Rename columns for clarity
        rename_dict = {
            'eventid_count': 'incident_count',
            'nkill_sum': 'total_deaths',
            'nwound_sum': 'total_wounded',
            'total_casualties_sum': 'total_casualties',
            'latitude_mean': 'latitude',
            'longitude_mean': 'longitude',
            'iyear_min': 'first_year',
            'iyear_max': 'last_year',
            'country_txt_first': 'country_txt',
            'region_txt_first': 'region_txt'
        }
        aggregated = aggregated.rename(columns=rename_dict)
        
        # Calculate year span
        aggregated['year_span'] = aggregated['last_year'] - aggregated['first_year'] + 1
        
        # Sort by total deaths descending
        aggregated = aggregated.sort_values('total_deaths', ascending=False)
        
        print(f"Aggregated to {len(aggregated)} unique locations")
        print(f"Top 5 deadliest locations:")
        top_5 = aggregated.head()[['city', 'country_txt', 'total_deaths', 'incident_count']].values
        for i, (city, country, deaths, incidents) in enumerate(top_5, 1):
            print(f"  {i}. {city}, {country}: {deaths:,} deaths in {incidents} incidents")
        
        return aggregated
    
    def filter_data_for_visualization(self, df, min_deaths=1, max_records=5000):
        """Filter data to optimize for visualization performance"""
        print(f"Filtering data for visualization (min_deaths={min_deaths}, max_records={max_records})...")
        
        # Filter by minimum deaths
        filtered_df = df[df['total_deaths'] >= min_deaths].copy()
        
        # If still too many records, take top N by deaths
        if len(filtered_df) > max_records:
            filtered_df = filtered_df.head(max_records)
            print(f"Limited to top {max_records} records by death count")
        
        print(f"Visualization dataset: {len(filtered_df):,} records")
        return filtered_df
    
    def get_summary_stats(self, df):
        """Get summary statistics for the processed data"""
        if df is None or len(df) == 0:
            return None
        
        stats = {
            'total_incidents': len(df),
            'total_deaths': df['nkill'].sum() if 'nkill' in df.columns else df['total_deaths'].sum(),
            'avg_deaths_per_incident': df['nkill'].mean() if 'nkill' in df.columns else df['total_deaths'].mean(),
            'deadliest_incident': df['nkill'].max() if 'nkill' in df.columns else df['total_deaths'].max(),
            'countries_affected': df['country_txt'].nunique(),
            'year_range': f"{df['iyear'].min()}-{df['iyear'].max()}" if 'iyear' in df.columns else f"{df['first_year'].min()}-{df['last_year'].max()}",
            'top_countries': df.groupby('country_txt')['nkill' if 'nkill' in df.columns else 'total_deaths'].sum().sort_values(ascending=False).head(5).to_dict()
        }
        
        return stats

if __name__ == "__main__":
    # Example usage
    from s3_data_loader import TerrorismDataLoader
    
    # Load data
    loader = TerrorismDataLoader()
    raw_data = loader.load_data()
    
    if raw_data is not None:
        # Process data
        processor = TerrorismDataProcessor()
        cleaned_data = processor.clean_data(raw_data)
        
        # Create aggregated data
        city_aggregated = processor.create_aggregated_data(cleaned_data, group_by='city')
        country_aggregated = processor.create_aggregated_data(cleaned_data, group_by='country')
        
        # Get summary stats
        stats = processor.get_summary_stats(cleaned_data)
        print("\nSummary Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Example: Fit linear regression
        # linear_results = processor.fit_linear_regression(cleaned_data)
        
        # Example: Fit LASSO regression
        # lasso_results = processor.fit_lasso_regression(cleaned_data)
        
        # Example: Visualize LASSO features
        # processor.visualize_lasso_features(top_n=20, save_path="lasso_features.png")
        
        # Example: Save LASSO features to CSV
        # processor.save_lasso_features("lasso_selected_features.csv")