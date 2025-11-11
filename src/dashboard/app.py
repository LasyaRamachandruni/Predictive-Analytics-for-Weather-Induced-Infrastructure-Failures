"""
Interactive Dash dashboard for infrastructure failure predictions.

Usage:
    python -m src.dashboard.app

Or:
    python src/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State
from plotly.subplots import make_subplots

# Try to load artifacts
DEFAULT_ARTIFACTS_PATH = Path("models/latest")


def load_artifacts(artifacts_path: Path = DEFAULT_ARTIFACTS_PATH) -> tuple[pd.DataFrame, Dict, Dict]:
    """Load predictions, metrics, and feature columns from artifacts."""
    predictions_df = pd.DataFrame()
    metrics_dict = {}
    feature_cols = {}
    
    try:
        # Load predictions
        predictions_path = artifacts_path / "predictions.csv"
        if predictions_path.exists():
            predictions_df = pd.read_csv(predictions_path, parse_dates=["timestamp"])
        
        # Load metrics
        metrics_path = artifacts_path / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics_dict = json.load(f)
        
        # Load feature columns
        features_path = artifacts_path / "feature_columns.json"
        if features_path.exists():
            with features_path.open("r", encoding="utf-8") as f:
                feature_cols = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load artifacts: {e}")
    
    return predictions_df, metrics_dict, feature_cols


# Load data
predictions_df, metrics_dict, feature_cols = load_artifacts()

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Infrastructure Failure Prediction Dashboard",
    suppress_callback_exceptions=True  # Allow callbacks to components created dynamically
)

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🏗️ Infrastructure Failure Prediction Dashboard", className="text-center mb-4"),
            html.Hr(),
        ])
    ]),
    
    # Navigation tabs
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="📊 Overview", tab_id="overview"),
                dbc.Tab(label="🗺️ Risk Map", tab_id="map"),
                dbc.Tab(label="📈 Predictions", tab_id="predictions"),
                dbc.Tab(label="📉 Model Performance", tab_id="performance"),
                dbc.Tab(label="🔍 Model Comparison", tab_id="comparison"),
            ], id="tabs", active_tab="overview"),
        ])
    ], className="mb-4"),
    
    # Tab content
    html.Div(id="tab-content"),
    
    # Footer
    html.Hr(),
    html.P("Infrastructure Failure Prediction System", className="text-center text-muted"),
    
], fluid=True)


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    """Render content based on active tab."""
    if active_tab == "overview":
        return render_overview()
    elif active_tab == "map":
        return render_map()
    elif active_tab == "predictions":
        return render_predictions()
    elif active_tab == "performance":
        return render_performance()
    elif active_tab == "comparison":
        return render_comparison()
    return html.Div("Select a tab")


def render_overview() -> html.Div:
    """Render overview dashboard."""
    if predictions_df.empty:
        return dbc.Alert("No data available. Please train a model first.", color="warning")
    
    # Calculate summary statistics
    total_predictions = len(predictions_df)
    num_regions = predictions_df["region_id"].nunique() if "region_id" in predictions_df.columns else 0
    date_range = f"{predictions_df['timestamp'].min()} to {predictions_df['timestamp'].max()}" if "timestamp" in predictions_df.columns else "N/A"
    
    avg_prediction = predictions_df["hybrid_pred"].mean() if "hybrid_pred" in predictions_df.columns else 0
    max_prediction = predictions_df["hybrid_pred"].max() if "hybrid_pred" in predictions_df.columns else 0
    
    # Get metrics summary
    test_metrics = metrics_dict.get("metrics", {}).get("test", {})
    hybrid_metrics = test_metrics.get("hybrid_ensemble", {})
    
    return dbc.Row([
        # Summary cards
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Predictions", className="card-title"),
                    html.H2(f"{total_predictions:,}", className="text-primary"),
                ])
            ], className="mb-3")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Regions", className="card-title"),
                    html.H2(f"{num_regions}", className="text-success"),
                ])
            ], className="mb-3")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Prediction", className="card-title"),
                    html.H2(f"{avg_prediction:.2f}", className="text-info"),
                ])
            ], className="mb-3")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Max Prediction", className="card-title"),
                    html.H2(f"{max_prediction:.2f}", className="text-danger"),
                ])
            ], className="mb-3")
        ], width=3),
        
        # Metrics card
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Performance (Test Set)"),
                dbc.CardBody([
                    html.Div([
                        html.P(f"RMSE: {hybrid_metrics.get('rmse', 'N/A'):.3f}" if isinstance(hybrid_metrics.get('rmse'), (int, float)) else "RMSE: N/A"),
                        html.P(f"MAE: {hybrid_metrics.get('mae', 'N/A'):.3f}" if isinstance(hybrid_metrics.get('mae'), (int, float)) else "MAE: N/A"),
                        html.P(f"R²: {hybrid_metrics.get('r2', 'N/A'):.3f}" if isinstance(hybrid_metrics.get('r2'), (int, float)) else "R²: N/A"),
                    ])
                ])
            ])
        ], width=12, className="mt-3"),
        
        # Time series plot
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Predictions Over Time"),
                dbc.CardBody([
                    dcc.Graph(id="overview-timeseries")
                ])
            ])
        ], width=12, className="mt-3"),
    ])


@app.callback(
    Output("overview-timeseries", "figure"),
    Input("tabs", "active_tab"),
    prevent_initial_call=True
)
def update_overview_timeseries(active_tab):
    """Update overview time series plot."""
    try:
        if predictions_df.empty or active_tab != "overview":
            return go.Figure()
        
        if "timestamp" not in predictions_df.columns or "hybrid_pred" not in predictions_df.columns:
            return go.Figure()
        
        # Aggregate by date
        daily = predictions_df.groupby("timestamp").agg({
            "hybrid_pred": "mean",
            "target": "mean" if "target" in predictions_df.columns else "first"
        }).reset_index()
        
        if daily.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        if "target" in daily.columns and daily["target"].notna().any():
            fig.add_trace(go.Scatter(
                x=daily["timestamp"],
                y=daily["target"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="blue", width=2)
            ))
        
        if "hybrid_pred" in daily.columns and daily["hybrid_pred"].notna().any():
            fig.add_trace(go.Scatter(
                x=daily["timestamp"],
                y=daily["hybrid_pred"],
                mode="lines+markers",
                name="Predicted",
                line=dict(color="red", width=2, dash="dash")
            ))
        
        fig.update_layout(
            title="Predictions vs Actual Over Time",
            xaxis_title="Date",
            yaxis_title="Failures",
            hovermode="x unified",
            height=400
        )
        
        return fig
    except Exception as e:
        print(f"Error in update_overview_timeseries: {e}")
        return go.Figure()


def render_map() -> html.Div:
    """Render interactive risk map."""
    if predictions_df.empty:
        return dbc.Alert("No data available. Please train a model first.", color="warning")
    
    if "latitude" not in predictions_df.columns or "longitude" not in predictions_df.columns:
        return dbc.Alert("Location data not available.", color="warning")
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    html.Label("Split:"),
                    dcc.Dropdown(
                        id="map-split-filter",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Train", "value": "train"},
                            {"label": "Validation", "value": "val"},
                            {"label": "Test", "value": "test"},
                        ],
                        value="test",
                        className="mb-3"
                    ),
                    html.Label("Region:"),
                    dcc.Dropdown(
                        id="map-region-filter",
                        options=[{"label": "All", "value": "all"}] + [
                            {"label": region, "value": region}
                            for region in sorted(predictions_df["region_id"].unique())
                        ] if "region_id" in predictions_df.columns else [],
                        value="all",
                    ),
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Risk Map"),
                dbc.CardBody([
                    dcc.Graph(id="risk-map")
                ])
            ])
        ], width=9),
    ])


@app.callback(
    Output("risk-map", "figure"),
    [Input("map-split-filter", "value"),
     Input("map-region-filter", "value"),
     Input("tabs", "active_tab")],
    prevent_initial_call=True
)
def update_risk_map(split_filter, region_filter, active_tab):
    """Update risk map based on filters."""
    try:
        if active_tab != "map":
            return go.Figure()
        if predictions_df.empty:
            return go.Figure()
        
        df = predictions_df.copy()
        
        # Apply filters
        if split_filter and split_filter != "all" and "split" in df.columns:
            df = df[df["split"] == split_filter]
        
        if region_filter and region_filter != "all" and "region_id" in df.columns:
            df = df[df["region_id"] == region_filter]
        
        if df.empty:
            return go.Figure()
        
        # Check for required columns
        required_cols = ["latitude", "longitude", "hybrid_pred"]
        if not all(col in df.columns for col in required_cols):
            return go.Figure()
        
        # Remove rows with missing coordinates
        df = df.dropna(subset=["latitude", "longitude", "hybrid_pred"])
        if df.empty:
            return go.Figure()
        
        # Aggregate by region
        if "region_id" in df.columns:
            df_agg = df.groupby("region_id").agg({
                "latitude": "first",
                "longitude": "first",
                "hybrid_pred": "mean",
            }).reset_index()
        else:
            df_agg = df[["latitude", "longitude", "hybrid_pred"]].copy()
        
        if df_agg.empty:
            return go.Figure()
        
        fig = px.scatter_geo(
            df_agg,
            lat="latitude",
            lon="longitude",
            size="hybrid_pred",
            color="hybrid_pred",
            hover_name="region_id" if "region_id" in df_agg.columns else None,
            hover_data={"hybrid_pred": ":.2f"},
            color_continuous_scale="Reds",
            title="Infrastructure Failure Risk Map",
            scope="usa"
        )
        
        fig.update_layout(
            height=600,
            geo=dict(
                projection_scale=3,
                center=dict(lat=39.5, lon=-98.35)
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error in update_risk_map: {e}")
        return go.Figure()


def render_predictions() -> html.Div:
    """Render predictions table and charts."""
    if predictions_df.empty:
        return dbc.Alert("No data available. Please train a model first.", color="warning")
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    html.Label("Split:"),
                    dcc.Dropdown(
                        id="pred-split-filter",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Train", "value": "train"},
                            {"label": "Validation", "value": "val"},
                            {"label": "Test", "value": "test"},
                        ],
                        value="all",
                        className="mb-3"
                    ),
                    html.Label("Region:"),
                    dcc.Dropdown(
                        id="pred-region-filter",
                        options=[{"label": "All", "value": "all"}] + [
                            {"label": region, "value": region}
                            for region in sorted(predictions_df["region_id"].unique())
                        ] if "region_id" in predictions_df.columns else [],
                        value="all",
                        className="mb-3"
                    ),
                    html.Label("Date Range:"),
                    dcc.DatePickerRange(
                        id="pred-date-range",
                        start_date=predictions_df["timestamp"].min() if "timestamp" in predictions_df.columns else None,
                        end_date=predictions_df["timestamp"].max() if "timestamp" in predictions_df.columns else None,
                    ),
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Predictions Table"),
                dbc.CardBody([
                    html.Div(id="predictions-table")
                ])
            ]),
            dbc.Card([
                dbc.CardHeader("Predictions Chart"),
                dbc.CardBody([
                    dcc.Graph(id="predictions-chart")
                ])
            ], className="mt-3")
        ], width=9),
    ])


@app.callback(
    [Output("predictions-table", "children"),
     Output("predictions-chart", "figure")],
    [Input("pred-split-filter", "value"),
     Input("pred-region-filter", "value"),
     Input("pred-date-range", "start_date"),
     Input("pred-date-range", "end_date"),
     Input("tabs", "active_tab")],
    prevent_initial_call=True
)
def update_predictions(split_filter, region_filter, start_date, end_date, active_tab):
    """Update predictions table and chart."""
    try:
        if active_tab != "predictions":
            return html.Div(""), go.Figure()
        if predictions_df.empty:
            return html.Div("No data"), go.Figure()
        
        df = predictions_df.copy()
        
        # Apply filters
        if split_filter and split_filter != "all" and "split" in df.columns:
            df = df[df["split"] == split_filter]
        
        if region_filter and region_filter != "all" and "region_id" in df.columns:
            df = df[df["region_id"] == region_filter]
        
        if start_date and "timestamp" in df.columns:
            try:
                df = df[df["timestamp"] >= pd.to_datetime(start_date)]
            except:
                pass
        
        if end_date and "timestamp" in df.columns:
            try:
                df = df[df["timestamp"] <= pd.to_datetime(end_date)]
            except:
                pass
        
        if df.empty:
            return html.Div("No data matching filters"), go.Figure()
        
        # Create table
        display_cols = ["timestamp", "region_id", "hybrid_pred", "target"]
        available_cols = [col for col in display_cols if col in df.columns]
        
        if not available_cols:
            return html.Div("No displayable columns"), go.Figure()
        
        table = dbc.Table.from_dataframe(
            df[available_cols].head(100),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="table-sm"
        )
        
        # Create chart
        fig = go.Figure()
        if "timestamp" in df.columns and "hybrid_pred" in df.columns:
            try:
                daily = df.groupby("timestamp").agg({
                    "hybrid_pred": "mean",
                    "target": "mean" if "target" in df.columns else "first"
                }).reset_index()
                
                if not daily.empty:
                    if "target" in daily.columns and daily["target"].notna().any():
                        fig.add_trace(go.Scatter(
                            x=daily["timestamp"],
                            y=daily["target"],
                            mode="lines+markers",
                            name="Actual",
                            line=dict(color="blue", width=2)
                        ))
                    
                    if "hybrid_pred" in daily.columns and daily["hybrid_pred"].notna().any():
                        fig.add_trace(go.Scatter(
                            x=daily["timestamp"],
                            y=daily["hybrid_pred"],
                            mode="lines+markers",
                            name="Predicted",
                            line=dict(color="red", width=2, dash="dash")
                        ))
                    
                    fig.update_layout(
                        title="Predictions Over Time",
                        xaxis_title="Date",
                        yaxis_title="Failures",
                        hovermode="x unified",
                        height=400
                    )
            except Exception as e:
                print(f"Error creating chart: {e}")
        
        return table, fig
    except Exception as e:
        print(f"Error in update_predictions: {e}")
        return html.Div(f"Error: {str(e)}"), go.Figure()


def render_performance() -> html.Div:
    """Render model performance metrics."""
    if not metrics_dict:
        return dbc.Alert("No metrics available. Please train a model first.", color="warning")
    
    metrics = metrics_dict.get("metrics", {})
    
    # Create performance comparison chart
    splits = ["train", "val", "test"]
    models = ["hybrid_ensemble", "lstm", "tabular_ensemble"]
    
    rmse_data = []
    mae_data = []
    r2_data = []
    
    for split in splits:
        split_metrics = metrics.get(split, {})
        for model in models:
            model_metrics = split_metrics.get(model, {})
            if isinstance(model_metrics.get("rmse"), (int, float)):
                rmse_data.append({
                    "Split": split.capitalize(),
                    "Model": model.replace("_", " ").title(),
                    "RMSE": model_metrics["rmse"]
                })
            if isinstance(model_metrics.get("mae"), (int, float)):
                mae_data.append({
                    "Split": split.capitalize(),
                    "Model": model.replace("_", " ").title(),
                    "MAE": model_metrics["mae"]
                })
            if isinstance(model_metrics.get("r2"), (int, float)):
                r2_data.append({
                    "Split": split.capitalize(),
                    "Model": model.replace("_", " ").title(),
                    "R²": model_metrics["r2"]
                })
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Performance Metrics"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=px.bar(
                            pd.DataFrame(rmse_data) if rmse_data else pd.DataFrame(),
                            x="Split",
                            y="RMSE",
                            color="Model",
                            barmode="group",
                            title="RMSE by Model and Split"
                        ) if rmse_data else go.Figure()
                    )
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Mean Absolute Error"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=px.bar(
                            pd.DataFrame(mae_data) if mae_data else pd.DataFrame(),
                            x="Split",
                            y="MAE",
                            color="Model",
                            barmode="group",
                            title="MAE by Model and Split"
                        ) if mae_data else go.Figure()
                    )
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("R² Score"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=px.bar(
                            pd.DataFrame(r2_data) if r2_data else pd.DataFrame(),
                            x="Split",
                            y="R²",
                            color="Model",
                            barmode="group",
                            title="R² by Model and Split"
                        ) if r2_data else go.Figure()
                    )
                ])
            ])
        ], width=12, className="mt-3"),
    ])


def render_comparison() -> html.Div:
    """Render model comparison charts."""
    try:
        if predictions_df.empty:
            return dbc.Alert("No data available. Please train a model first.", color="warning")
        
        # Create scatter plot comparing models
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Actual vs Predicted (Hybrid)", "Model Comparison"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if "target" in predictions_df.columns and "hybrid_pred" in predictions_df.columns:
            # Filter out NaN values
            valid_mask = predictions_df["target"].notna() & predictions_df["hybrid_pred"].notna()
            if valid_mask.any():
                # Actual vs Predicted
                fig.add_trace(
                    go.Scatter(
                        x=predictions_df.loc[valid_mask, "target"],
                        y=predictions_df.loc[valid_mask, "hybrid_pred"],
                        mode="markers",
                        name="Hybrid",
                        marker=dict(color="blue", opacity=0.6)
                    ),
                    row=1, col=1
                )
                
                # Add diagonal line
                min_val = min(predictions_df.loc[valid_mask, "target"].min(), 
                             predictions_df.loc[valid_mask, "hybrid_pred"].min())
                max_val = max(predictions_df.loc[valid_mask, "target"].max(), 
                             predictions_df.loc[valid_mask, "hybrid_pred"].max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="Perfect",
                        line=dict(color="red", dash="dash")
                    ),
                    row=1, col=1
                )
                
                fig.update_xaxes(title_text="Actual", row=1, col=1)
                fig.update_yaxes(title_text="Predicted", row=1, col=1)
        
        # Model comparison
        if "lstm_pred" in predictions_df.columns and "tabular_ensemble" in predictions_df.columns:
            valid_mask = predictions_df["lstm_pred"].notna() & predictions_df["tabular_ensemble"].notna()
            if valid_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=predictions_df.loc[valid_mask, "lstm_pred"],
                        y=predictions_df.loc[valid_mask, "tabular_ensemble"],
                        mode="markers",
                        name="LSTM vs Tabular",
                        marker=dict(color="green", opacity=0.6)
                    ),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="LSTM Prediction", row=1, col=2)
                fig.update_yaxes(title_text="Tabular Prediction", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=True)
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Comparison"),
                    dbc.CardBody([
                        dcc.Graph(figure=fig)
                    ])
                ])
            ], width=12)
        ])
    except Exception as e:
        print(f"Error in render_comparison: {e}")
        return dbc.Alert(f"Error rendering comparison: {str(e)}", color="danger")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Dashboard Starting...")
    print("=" * 60)
    print("📊 Open your browser and go to:")
    print("   http://localhost:8050")
    print("   or")
    print("   http://127.0.0.1:8050")
    print("=" * 60)
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=True, host="127.0.0.1", port=8050)

