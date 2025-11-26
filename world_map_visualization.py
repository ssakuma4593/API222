"""
World Map Visualization of Terrorist Attacks
Creates an interactive world map with bubbles showing attack locations
and bubble size representing attack severity
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def load_and_prepare_data(csv_path='data/gtd_visualization.csv'):
    """Load and prepare data for visualization.

    Expects a slim CSV created by
    ``TerrorismDataProcessor.create_visualization_dataset_from_raw()``
    with (at least) the columns:
    eventid, iyear, city, country_txt, latitude, longitude,
    severity, gname, attacktype1_txt.
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df):,} records")
    
    # Filter out records without valid coordinates
    initial_count = len(df)
    df = df.dropna(subset=['latitude', 'longitude', 'severity'])
    
    # Filter out invalid coordinates
    df = df[
        (df['latitude'] >= -90) & (df['latitude'] <= 90) &
        (df['longitude'] >= -180) & (df['longitude'] <= 180)
    ]
    
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Removed {removed:,} records with missing or invalid coordinates/severity")
    
    # Ensure severity is numeric and non-negative
    df['severity'] = pd.to_numeric(df['severity'], errors='coerce')
    df = df[df['severity'] >= 0]
    
    # Fill missing text fields
    text_columns = ['city', 'gname', 'country_txt', 'attacktype1_txt']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    print(f"Final dataset: {len(df):,} records")
    print(f"Severity range: {df['severity'].min():.0f} - {df['severity'].max():.0f}")
    print(f"Countries: {df['country_txt'].nunique()}")
    
    return df

def create_world_map(df, output_file='world_map_attacks.html', 
                     max_points=5000, min_severity=0):
    """
    Create an interactive world map with bubbles showing attack locations
    
    Parameters:
    - df: DataFrame with latitude, longitude, and severity columns
    - output_file: Output HTML file name
    - max_points: Maximum number of points to display (for performance)
    - min_severity: Minimum severity to display
    """
    print("\nCreating world map visualization...")
    
    # Filter by minimum severity
    if min_severity > 0:
        df = df[df['severity'] >= min_severity].copy()
        print(f"Filtered to {len(df):,} records with severity >= {min_severity}")
    
    # If too many points, sample or aggregate
    if len(df) > max_points:
        print(f"Too many points ({len(df):,}). Sampling top {max_points} by severity...")
        df = df.nlargest(max_points, 'severity')
    
    # Create hover text with attack details
    hover_data = []
    for idx, row in df.iterrows():
        city = str(row.get('city', 'Unknown'))
        country = str(row.get('country_txt', 'Unknown'))
        hover_text = f"<b>Location:</b> {city}, {country}<br>"
        hover_text += f"<b>Severity:</b> {row['severity']:.0f}<br>"
        if pd.notna(row.get('iyear')):
            hover_text += f"<b>Year:</b> {int(row['iyear'])}<br>"
        if pd.notna(row.get('gname')) and str(row.get('gname')) not in ('Unknown', 'nan'):
            hover_text += f"<b>Group:</b> {row['gname']}<br>"
        # Use attacktype1_txt directly if available
        attack_type = row.get('attacktype1_txt')
        if pd.notna(attack_type) and str(attack_type) not in ('Unknown', 'nan'):
            hover_text += f"<b>Attack Type:</b> {attack_type}"
        hover_data.append(hover_text)
    
    df['hover_text'] = hover_data
    
    # Normalize bubble sizes (scale between 2 and 12 pixels - much smaller)
    max_severity = df['severity'].max()
    min_severity = df['severity'].min()
    
    if max_severity > min_severity:
        # Scale severity to bubble size (2-12 pixels - smaller bubbles)
        df['bubble_size'] = 2 + (df['severity'] - min_severity) / (max_severity - min_severity) * 10
    else:
        df['bubble_size'] = 5  # Default size if all severities are the same
    
    # Create the map using plotly express for better map rendering
    fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        size='severity',
        color='severity',
        hover_name='city',
        hover_data={
            'country_txt': True,
            'iyear': True,
            'severity': ':.0f',
            'latitude': False,
            'longitude': False
        },
        size_max=15,  # Maximum bubble size in pixels
        color_continuous_scale='YlOrRd',
        title='<b>Global Terrorism Database: Attack Locations by Severity</b><br>' +
              f'<sub>Showing {len(df):,} attacks | Bubble size and color indicate severity</sub>',
        projection='natural earth'
    )
    
    # Update marker properties for better visibility
    fig.update_traces(
        marker=dict(
            line=dict(width=0.3, color='darkgray'),
            opacity=0.6,
            sizemode='diameter',
            sizemin=2
        )
    )
    
    # Update layout for better map visibility
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showocean=True,
            oceancolor='rgb(204, 229, 255)',
            showlakes=True,
            lakecolor='rgb(204, 229, 255)',
            showcountries=True,
            countrycolor='rgb(200, 200, 200)',
            coastlinecolor='rgb(200, 200, 200)',
            showframe=True,
            framecolor='rgb(100, 100, 100)',
            bgcolor='rgb(230, 240, 255)'
        ),
        height=800,
        margin=dict(l=0, r=0, t=80, b=0),
        coloraxis_colorbar=dict(
            title="Severity",
            x=1.02
        )
    )
    
    # Save the map
    print(f"Saving map to {output_file}...")
    fig.write_html(output_file)
    print(f"Map saved successfully!")
    
    return fig

def create_summary_statistics(df):
    """Print summary statistics about the attacks"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total attacks: {len(df):,}")
    print(f"Total severity: {df['severity'].sum():,.0f}")
    print(f"Average severity per attack: {df['severity'].mean():.2f}")
    print(f"Median severity: {df['severity'].median():.0f}")
    print(f"Max severity: {df['severity'].max():.0f}")
    print(f"Min severity: {df['severity'].min():.0f}")
    
    if 'country_txt' in df.columns:
        print(f"\nTop 10 countries by total severity:")
        top_countries = df.groupby('country_txt')['severity'].sum().sort_values(ascending=False).head(10)
        for i, (country, severity) in enumerate(top_countries.items(), 1):
            print(f"  {i:2d}. {country}: {severity:,.0f}")
    
    if 'iyear' in df.columns:
        print(f"\nYear range: {int(df['iyear'].min())} - {int(df['iyear'].max())}")
    
    print("="*60)

if __name__ == "__main__":
    # Load data
    df = load_and_prepare_data()
    
    # Print summary statistics
    create_summary_statistics(df)
    
    # Create the world map
    # Adjust max_points and min_severity as needed for performance/clarity
    fig = create_world_map(
        df, 
        output_file='world_map_attacks.html',
        max_points=5000,  # Limit for performance
        min_severity=0    # Show all attacks
    )
    
    print("\nVisualization complete!")
    print("Open 'world_map_attacks.html' in your web browser to view the interactive map.")

