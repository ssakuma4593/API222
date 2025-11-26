"""
World Map Visualization of Terrorist Attacks
Creates an interactive world map with bubbles showing attack locations
and bubble size representing attack severity.

This version also supports highlighting events that match the top
LASSO-selected features via an interactive dropdown (e.g., suicides,
specific weapon types, regions, or countries).
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


def load_top_lasso_features(path="lasso_selected_features.csv", top_n=None):
    """Load positive LASSO coefficients from CSV.

    Assumes the CSV was created by regression_models.save_lasso_features()
    and has a single column of coefficients, with the index being the
    feature names.
    """
    try:
        # CSV saved by regression_models.save_lasso_features() has
        # feature names as the first column (index) and a single column
        # of coefficients.
        df = pd.read_csv(path, index_col=0)
        # Take the first (and only) column as a Series: index = feature,
        # value = coefficient
        s = df.iloc[:, 0]
    except Exception:
        print(f"Warning: could not load LASSO features from {path}")
        return []

    # Keep only positively associated features (coef > 0)
    s = s[s > 0]

    # Sort by coefficient magnitude, descending
    s = s.sort_values(ascending=False)

    if top_n is not None:
        s = s.head(top_n)

    print("Positive LASSO features used for highlighting:")
    for feat, coef in s.items():
        print(f"  {feat}: {coef:.4f}")
    return s


def _mask_for_lasso_feature(df, feature_name):
    """Return a boolean mask for rows matching a given LASSO feature."""
    # Simple binary columns
    if feature_name in ["suicide", "success"]:
        if feature_name in df.columns:
            return df[feature_name] == 1
        return pd.Series(False, index=df.index)

    # Prefix-based categorical encodings from regression_models
    prefix_map = {
        "country_txt_": "country_txt",
        "region_txt_": "region_txt",
        "attacktype1_txt_": "attacktype1_txt",
        "weaptype1_txt_": "weaptype1_txt",
        "targtype1_txt_": "targtype1_txt",
    }
    for prefix, col in prefix_map.items():
        if feature_name.startswith(prefix) and col in df.columns:
            value = feature_name[len(prefix):]
            return df[col] == value

    # Fallback: if the feature name is a numeric column, highlight values
    # above the median (rare in our current setup, but safe).
    if feature_name in df.columns and np.issubdtype(df[feature_name].dtype, np.number):
        median_val = df[feature_name].median()
        return df[feature_name] >= median_val

    # Default: no matches
    return pd.Series(False, index=df.index)


def create_world_map(df, output_file='world_map_attacks.html',
                     max_points=5000, min_severity=0,
                     top_lasso_features=None):
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

        # Date information
        year = row.get("iyear")
        month = row.get("imonth")
        day = row.get("iday")
        if pd.notna(year):
            if pd.notna(month) and pd.notna(day):
                hover_text += f"<b>Date:</b> {int(year)}-{int(month):02d}-{int(day):02d}<br>"
            else:
                hover_text += f"<b>Year:</b> {int(year)}<br>"

        # Region
        region = row.get("region_txt")
        if pd.notna(region) and str(region) not in ("Unknown", "nan"):
            hover_text += f"<b>Region:</b> {region}<br>"

        if pd.notna(row.get('gname')) and str(row.get('gname')) not in ('Unknown', 'nan'):
            hover_text += f"<b>Group:</b> {row['gname']}<br>"

        # Use attacktype1_txt, weaptype1_txt, targtype1_txt directly if available
        attack_type = row.get('attacktype1_txt')
        if pd.notna(attack_type) and str(attack_type) not in ('Unknown', 'nan'):
            hover_text += f"<b>Attack Type:</b> {attack_type}<br>"

        weapon_type = row.get("weaptype1_txt")
        if pd.notna(weapon_type) and str(weapon_type) not in ("Unknown", "nan"):
            hover_text += f"<b>Weapon Type:</b> {weapon_type}<br>"

        target_type = row.get("targtype1_txt")
        if pd.notna(target_type) and str(target_type) not in ("Unknown", "nan"):
            hover_text += f"<b>Target Type:</b> {target_type}<br>"

        # Success / suicide flags
        success = row.get("success")
        if pd.notna(success):
            hover_text += f"<b>Successful:</b> {bool(success)}<br>"

        suicide = row.get("suicide")
        if pd.notna(suicide):
            hover_text += f"<b>Suicide attack:</b> {bool(suicide)}<br>"
        hover_data.append(hover_text)
    
    df['hover_text'] = hover_data
    
    # Create the map using plotly express for better map rendering.
    # We use severity both for color and size (as in the original map),
    # so bubble size always reflects attack severity.
    # Use a slightly darker custom YlOrRd-like scale so very low
    # severity points are still visible (avoid near-white yellows).
    darker_ylorrd = [
        [0.0,  "#FEE090"],
        [0.2,  "#FDC863"],
        [0.4,  "#F9A602"],
        [0.6,  "#F46D43"],
        [0.8,  "#D73027"],
        [1.0,  "#7F0000"],
    ]

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
        color_continuous_scale=darker_ylorrd,
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
    
    # Optionally add a dropdown to filter by top LASSO features.
    # Behaviour:
    # - Initial view: original map with all attacks (size and color by severity).
    # - When a feature is selected: hide the base trace and show only the
    #   subset of attacks matching that feature, using the same size and
    #   color mapping.
    if top_lasso_features is not None and not top_lasso_features.empty:
        feature_traces = []
        for feat, coef in top_lasso_features.items():
            mask = _mask_for_lasso_feature(df, feat)
            sub = df[mask].copy()
            if sub.empty:
                continue
            label = f"{feat} ({coef:.2f})"
            feature_traces.append(
                go.Scattergeo(
                    lat=sub["latitude"],
                    lon=sub["longitude"],
                    mode="markers",
                    marker=dict(
                        size=sub["severity"],
                        color=sub["severity"],
                        colorscale=darker_ylorrd,
                        cmin=df["severity"].min(),
                        cmax=df["severity"].max(),
                        line=dict(width=0.3, color="darkgray"),
                        sizemode="diameter",
                        sizemin=2,
                    ),
                    name=f"LASSO: {label}",
                    hovertext=sub["hover_text"],
                    hoverinfo="text",
                    visible=False,
                    showlegend=False,
                )
            )

        for trace in feature_traces:
            fig.add_trace(trace)

        n_extra = len(feature_traces)
        if n_extra:
            buttons = []

            # "All attacks" button: show only the original full dataset
            visible_all = [True] + [False] * n_extra
            buttons.append(
                dict(
                    label="All attacks",
                    method="update",
                    args=[{"visible": visible_all},
                          {"title": fig.layout.title.text}],
                )
            )

            # One button per feature trace: hide base, show only that trace
            for i, trace in enumerate(feature_traces, start=1):
                visible = [False] * (n_extra + 1)
                visible[i] = True
                buttons.append(
                    dict(
                        label=trace.name,
                        method="update",
                        args=[
                            {"visible": visible},
                            {
                                "title": fig.layout.title.text
                                + f"<br><sub>Filter: {trace.name}</sub>"
                            },
                        ],
                    )
                )

            fig.update_layout(
                updatemenus=[
                    dict(
                        type="dropdown",
                        x=0.0,
                        y=1.15,
                        xanchor="left",
                        yanchor="top",
                        showactive=True,
                        buttons=buttons,
                        bgcolor="white",
                        bordercolor="gray",
                    )
                ]
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
    top_features = load_top_lasso_features("lasso_selected_features.csv", top_n=20)
    
    # Print summary statistics
    create_summary_statistics(df)
    
    # Create the world map
    # Adjust max_points and min_severity as needed for performance/clarity
    fig = create_world_map(
        df,
        output_file='world_map_attacks.html',
        max_points=5000,  # Limit for performance
        min_severity=0,   # Show all attacks
        top_lasso_features=top_features,
    )
    
    print("\nVisualization complete!")
    print("Open 'world_map_attacks.html' in your web browser to view the interactive map.")

