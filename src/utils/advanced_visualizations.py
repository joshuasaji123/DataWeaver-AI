"""
Advanced Visualization Tools for Multi-Agent Data Analysis

This module provides enhanced visualization capabilities including:
- Interactive dashboards
- Advanced statistical plots
- Domain-specific visualizations
- 3D and animated charts
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizationEngine:
    """
    Advanced visualization engine with multiple chart types and interactive features.
    """
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.theme = "plotly_white"
    
    def create_financial_dashboard(self, data: pd.DataFrame, financial_metrics: Dict) -> go.Figure:
        """
        Create an interactive financial dashboard with key metrics.
        
        Args:
            data: Financial data DataFrame
            financial_metrics: Dictionary of calculated financial metrics
            
        Returns:
            Plotly figure with financial dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Revenue Trends", "Profitability Ratios", "Cash Flow", "Key Metrics"),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Revenue trends (if time-based data exists)
        if 'date' in data.columns or 'period' in data.columns:
            date_col = 'date' if 'date' in data.columns else 'period'
            if 'revenue' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data[date_col], 
                        y=data['revenue'],
                        mode='lines+markers',
                        name='Revenue',
                        line=dict(color='#2E86AB', width=3)
                    ),
                    row=1, col=1
                )
        
        # Profitability ratios
        if financial_metrics:
            metrics_names = list(financial_metrics.keys())[:5]
            metrics_values = [financial_metrics[name] for name in metrics_names]
            
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=metrics_values,
                    name='Financial Ratios',
                    marker_color=['#A23B72', '#F18F01', '#C73E1D', '#2E86AB', '#0077B6']
                ),
                row=1, col=2
            )
        
        # Cash flow visualization
        if 'cash_flow' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['cash_flow'].cumsum(),
                    mode='lines',
                    name='Cumulative Cash Flow',
                    fill='tonexty',
                    line=dict(color='#0077B6')
                ),
                row=2, col=1
            )
        
        # Key metric indicator
        if 'roi' in financial_metrics:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=financial_metrics['roi'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "ROI (%)"},
                    delta={'reference': 15},
                    gauge={
                        'axis': {'range': [None, 50]},
                        'bar': {'color': "#2E86AB"},
                        'steps': [
                            {'range': [0, 10], 'color': "#ffcccb"},
                            {'range': [10, 25], 'color': "#fff3cd"},
                            {'range': [25, 50], 'color': "#d4edda"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 20
                        }
                    }
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Financial Performance Dashboard",
            showlegend=True,
            template=self.theme,
            height=600
        )
        
        return fig
    
    def create_sports_performance_chart(self, data: pd.DataFrame, player_col: str, metric_cols: List[str]) -> go.Figure:
        """
        Create sports performance visualization with player comparisons.
        
        Args:
            data: Sports data DataFrame
            player_col: Column name for player identification
            metric_cols: List of performance metric columns
            
        Returns:
            Plotly figure with sports performance analysis
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Performance Radar", "Metric Trends", "Player Comparison", "Performance Distribution"),
            specs=[[{"type": "scatterpolar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Performance radar chart for top player
        if len(metric_cols) >= 3:
            top_player_data = data.loc[data[metric_cols].sum(axis=1).idxmax()]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=[top_player_data[col] for col in metric_cols],
                    theta=metric_cols,
                    fill='toself',
                    name=f'{top_player_data[player_col]} Performance',
                    line_color='#2E86AB'
                ),
                row=1, col=1
            )
        
        # Metric trends over time (if applicable)
        if 'game_number' in data.columns or 'date' in data.columns:
            time_col = 'game_number' if 'game_number' in data.columns else 'date'
            main_metric = metric_cols[0] if metric_cols else 'points'
            
            fig.add_trace(
                go.Scatter(
                    x=data[time_col],
                    y=data[main_metric],
                    mode='lines+markers',
                    name=f'{main_metric.title()} Trend',
                    line=dict(color='#F18F01', width=2)
                ),
                row=1, col=2
            )
        
        # Player comparison (top 10 players)
        if len(data) > 1:
            top_players = data.nlargest(10, metric_cols[0] if metric_cols else 'points')
            
            fig.add_trace(
                go.Bar(
                    x=top_players[player_col],
                    y=top_players[metric_cols[0] if metric_cols else 'points'],
                    name='Top Performers',
                    marker_color='#A23B72'
                ),
                row=2, col=1
            )
        
        # Performance distribution
        if metric_cols:
            fig.add_trace(
                go.Box(
                    y=data[metric_cols[0]],
                    name=f'{metric_cols[0].title()} Distribution',
                    marker_color='#0077B6'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Sports Performance Analysis Dashboard",
            showlegend=True,
            template=self.theme,
            height=700
        )
        
        return fig
    
    def create_correlation_heatmap_advanced(self, data: pd.DataFrame, method: str = 'pearson') -> go.Figure:
        """
        Create an advanced correlation heatmap with clustering and annotations.
        
        Args:
            data: DataFrame with numeric columns
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Plotly heatmap figure
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return go.Figure().add_annotation(text="No numeric data available for correlation analysis")
        
        corr_matrix = numeric_data.corr(method=method)
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Correlation Heatmap ({method.title()})',
            template=self.theme,
            width=600,
            height=600
        )
        
        return fig
    
    def create_3d_scatter(self, data: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                         color_col: Optional[str] = None, size_col: Optional[str] = None) -> go.Figure:
        """
        Create a 3D scatter plot with optional color and size encoding.
        
        Args:
            data: DataFrame with the data
            x_col, y_col, z_col: Column names for x, y, z axes
            color_col: Optional column for color encoding
            size_col: Optional column for size encoding
            
        Returns:
            Plotly 3D scatter figure
        """
        fig = go.Figure()
        
        scatter_kwargs = {
            'x': data[x_col],
            'y': data[y_col],
            'z': data[z_col],
            'mode': 'markers',
            'marker': dict(
                size=8 if size_col is None else data[size_col],
                opacity=0.7,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            'text': data.index,
            'hovertemplate': f'<b>%{{text}}</b><br>' +
                           f'{x_col}: %{{x}}<br>' +
                           f'{y_col}: %{{y}}<br>' +
                           f'{z_col}: %{{z}}<extra></extra>'
        }
        
        if color_col:
            scatter_kwargs['marker']['color'] = data[color_col]
            scatter_kwargs['marker']['colorscale'] = 'Viridis'
            scatter_kwargs['marker']['showscale'] = True
        
        fig.add_trace(go.Scatter3d(**scatter_kwargs))
        
        fig.update_layout(
            title=f'3D Analysis: {x_col} vs {y_col} vs {z_col}',
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            template=self.theme,
            height=600
        )
        
        return fig
    
    def create_time_series_decomposition(self, data: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
        """
        Create a time series decomposition visualization.
        
        Args:
            data: DataFrame with time series data
            date_col: Column name for dates
            value_col: Column name for values
            
        Returns:
            Plotly figure with decomposition
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Prepare data
            ts_data = data.set_index(date_col)[value_col].dropna()
            
            # Perform decomposition
            decomposition = seasonal_decompose(ts_data, period=12 if len(ts_data) > 24 else len(ts_data)//2)
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=("Original", "Trend", "Seasonal", "Residual"),
                vertical_spacing=0.1
            )
            
            # Original data
            fig.add_trace(
                go.Scatter(x=ts_data.index, y=ts_data.values, name="Original", line=dict(color='#2E86AB')),
                row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name="Trend", line=dict(color='#F18F01')),
                row=2, col=1
            )
            
            # Seasonal
            fig.add_trace(
                go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name="Seasonal", line=dict(color='#A23B72')),
                row=3, col=1
            )
            
            # Residual
            fig.add_trace(
                go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name="Residual", line=dict(color='#0077B6')),
                row=4, col=1
            )
            
            fig.update_layout(
                title="Time Series Decomposition",
                template=self.theme,
                height=800,
                showlegend=False
            )
            
            return fig
            
        except ImportError:
            return go.Figure().add_annotation(text="statsmodels required for time series decomposition")
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error in decomposition: {str(e)}")
    
    def create_distribution_comparison(self, data: pd.DataFrame, columns: List[str], 
                                     chart_type: str = 'violin') -> go.Figure:
        """
        Create distribution comparison plots.
        
        Args:
            data: DataFrame with the data
            columns: List of columns to compare
            chart_type: Type of chart ('violin', 'box', 'histogram')
            
        Returns:
            Plotly figure with distribution comparison
        """
        if chart_type == 'violin':
            fig = go.Figure()
            
            for i, col in enumerate(columns):
                fig.add_trace(go.Violin(
                    y=data[col].dropna(),
                    name=col,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=self.color_palette[i % len(self.color_palette)],
                    opacity=0.6
                ))
                
        elif chart_type == 'box':
            fig = go.Figure()
            
            for i, col in enumerate(columns):
                fig.add_trace(go.Box(
                    y=data[col].dropna(),
                    name=col,
                    marker_color=self.color_palette[i % len(self.color_palette)]
                ))
                
        else:  # histogram
            fig = make_subplots(
                rows=len(columns), cols=1,
                subplot_titles=columns,
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(columns):
                fig.add_trace(
                    go.Histogram(
                        x=data[col].dropna(),
                        name=col,
                        marker_color=self.color_palette[i % len(self.color_palette)],
                        opacity=0.7
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title=f'Distribution Comparison ({chart_type.title()})',
            template=self.theme,
            height=400 * len(columns) if chart_type == 'histogram' else 500
        )
        
        return fig
    
    def create_wordcloud_visualization(self, text_data: List[str], title: str = "Word Cloud") -> go.Figure:
        """
        Create a word cloud visualization.
        
        Args:
            text_data: List of text strings
            title: Title for the visualization
            
        Returns:
            Plotly figure with word cloud
        """
        try:
            # Combine all text
            combined_text = ' '.join(text_data)
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis'
            ).generate(combined_text)
            
            # Convert to image
            fig = go.Figure()
            fig.add_layout_image(
                dict(
                    source=wordcloud.to_image(),
                    xref="x",
                    yref="y",
                    x=0,
                    y=0,
                    sizex=1,
                    sizey=1,
                    sizing="stretch",
                    opacity=1,
                    layer="below"
                )
            )
            
            fig.update_layout(
                title=title,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                template=self.theme,
                height=400
            )
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error creating word cloud: {str(e)}")
    
    def create_advanced_scatter_matrix(self, data: pd.DataFrame, dimensions: List[str], 
                                     color_col: Optional[str] = None) -> go.Figure:
        """
        Create an advanced scatter plot matrix with correlation information.
        
        Args:
            data: DataFrame with the data
            dimensions: List of column names to include
            color_col: Optional column for color encoding
            
        Returns:
            Plotly scatter matrix figure
        """
        # Filter to only include specified dimensions
        plot_data = data[dimensions].dropna()
        
        fig = go.Figure(data=go.Splom(
            dimensions=[dict(label=col, values=plot_data[col]) for col in dimensions],
            text=plot_data.index,
            marker=dict(
                color=plot_data[color_col] if color_col and color_col in plot_data.columns else '#2E86AB',
                showscale=True if color_col else False,
                line_color='white',
                line_width=0.5,
                colorscale='Viridis'
            )
        ))
        
        fig.update_layout(
            title='Advanced Scatter Plot Matrix',
            template=self.theme,
            height=600,
            width=800
        )
        
        return fig

# Utility functions for chart recommendations
def recommend_visualizations(data: pd.DataFrame, agent_role: str) -> List[Dict]:
    """
    Recommend appropriate visualizations based on data characteristics and agent role.
    
    Args:
        data: DataFrame to analyze
        agent_role: Role of the agent requesting recommendations
        
    Returns:
        List of visualization recommendations
    """
    recommendations = []
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Role-specific recommendations
    if agent_role == "Accountant":
        recommendations.extend([
            {"type": "financial_dashboard", "priority": "high", "description": "Financial performance overview"},
            {"type": "ratio_trends", "priority": "high", "description": "Financial ratio trends over time"},
            {"type": "cash_flow_analysis", "priority": "medium", "description": "Cash flow visualization"}
        ])
    
    elif agent_role == "Sports Coach":
        recommendations.extend([
            {"type": "performance_radar", "priority": "high", "description": "Player performance radar chart"},
            {"type": "team_comparison", "priority": "high", "description": "Team performance comparison"},
            {"type": "trend_analysis", "priority": "medium", "description": "Performance trends over time"}
        ])
    
    elif agent_role == "Data Scientist":
        recommendations.extend([
            {"type": "correlation_matrix", "priority": "high", "description": "Advanced correlation analysis"},
            {"type": "3d_scatter", "priority": "medium", "description": "Multi-dimensional relationships"},
            {"type": "distribution_analysis", "priority": "medium", "description": "Statistical distributions"}
        ])
    
    # Data-driven recommendations
    if len(numeric_cols) >= 3:
        recommendations.append({
            "type": "3d_scatter", 
            "priority": "medium", 
            "description": f"3D analysis of {numeric_cols[:3]}"
        })
    
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        recommendations.append({
            "type": "time_series", 
            "priority": "high", 
            "description": "Time series analysis and decomposition"
        })
    
    if len(categorical_cols) > 0:
        recommendations.append({
            "type": "category_analysis", 
            "priority": "medium", 
            "description": "Categorical data distribution analysis"
        })
    
    return recommendations 