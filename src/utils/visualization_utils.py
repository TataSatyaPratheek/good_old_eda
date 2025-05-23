"""
Visualization Utilities for SEO Competitive Intelligence
Advanced charting and dashboard creation utilities for SEO analysis and reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ChartGenerator:
    """
    Advanced chart generation for SEO competitive intelligence.
    
    Provides comprehensive visualization capabilities for SEO metrics,
    competitive analysis, and performance tracking.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Set default styling
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Color schemes for competitors
        self.competitor_colors = {
            'lenovo': '#E31837',
            'hp': '#0096D6', 
            'dell': '#007DB8',
            'microsoft': '#00BCF2',
            'asus': '#000000',
            'intel': '#0071C5'
        }
        
        # Chart configuration
        self.chart_config = {
            'figure_size': (12, 8),
            'dpi': 300,
            'font_size': 12,
            'title_size': 16,
            'legend_size': 10
        }

    def create_position_tracking_chart(
        self,
        position_data: pd.DataFrame,
        keywords: List[str] = None,
        chart_type: str = 'line',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create position tracking visualization.
        
        Args:
            position_data: DataFrame with position data over time
            keywords: Specific keywords to visualize
            chart_type: Type of chart ('line', 'heatmap', 'area')
            save_path: Optional path to save chart
            
        Returns:
            Plotly figure object
        """
        try:
            if keywords is None:
                # Select top keywords by traffic or volume
                if 'Traffic' in position_data.columns:
                    top_keywords = (position_data.groupby('Keyword')['Traffic']
                                  .sum().nlargest(10).index.tolist())
                else:
                    top_keywords = position_data['Keyword'].value_counts().head(10).index.tolist()
            else:
                top_keywords = keywords
            
            # Filter data
            filtered_data = position_data[position_data['Keyword'].isin(top_keywords)]
            
            if chart_type == 'line':
                fig = self._create_position_line_chart(filtered_data, top_keywords)
            elif chart_type == 'heatmap':
                fig = self._create_position_heatmap(filtered_data, top_keywords)
            elif chart_type == 'area':
                fig = self._create_position_area_chart(filtered_data, top_keywords)
            else:
                fig = self._create_position_line_chart(filtered_data, top_keywords)
            
            # Update layout
            fig.update_layout(
                title="Keyword Position Tracking Over Time",
                xaxis_title="Date",
                yaxis_title="Position",
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Invert y-axis (lower position number = better)
            fig.update_yaxis(autorange="reversed")
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Position tracking chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating position tracking chart: {str(e)}")
            return go.Figure()

    def create_competitive_landscape_chart(
        self,
        competitor_data: Dict[str, pd.DataFrame],
        metric: str = 'Traffic',
        visualization_type: str = 'scatter'
    ) -> go.Figure:
        """
        Create competitive landscape visualization.
        
        Args:
            competitor_data: Dictionary of competitor DataFrames
            metric: Metric to visualize
            visualization_type: Type of visualization
            
        Returns:
            Plotly figure object
        """
        try:
            # Combine competitor data
            combined_data = []
            for competitor, df in competitor_data.items():
                df_copy = df.copy()
                df_copy['competitor'] = competitor
                combined_data.append(df_copy)
            
            if not combined_data:
                return go.Figure()
            
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            if visualization_type == 'scatter':
                fig = self._create_competitive_scatter(combined_df, metric)
            elif visualization_type == 'bubble':
                fig = self._create_competitive_bubble(combined_df, metric)
            elif visualization_type == 'radar':
                fig = self._create_competitive_radar(competitor_data)
            else:
                fig = self._create_competitive_bar(combined_df, metric)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating competitive landscape chart: {str(e)}")
            return go.Figure()

    def create_traffic_funnel_chart(
        self,
        traffic_data: pd.DataFrame,
        position_ranges: List[Tuple[int, int]] = None
    ) -> go.Figure:
        """
        Create traffic distribution funnel by position ranges.
        
        Args:
            traffic_data: DataFrame with position and traffic data
            position_ranges: List of position ranges to analyze
            
        Returns:
            Plotly funnel chart
        """
        try:
            if position_ranges is None:
                position_ranges = [(1, 3), (4, 10), (11, 20), (21, 50), (51, 100)]
            
            # Calculate traffic by position ranges
            funnel_data = []
            
            for start_pos, end_pos in position_ranges:
                range_mask = (
                    (traffic_data['Position'] >= start_pos) & 
                    (traffic_data['Position'] <= end_pos)
                )
                range_traffic = traffic_data[range_mask]['Traffic'].sum()
                range_keywords = range_mask.sum()
                
                funnel_data.append({
                    'range': f"Positions {start_pos}-{end_pos}",
                    'traffic': range_traffic,
                    'keywords': range_keywords,
                    'avg_traffic_per_keyword': range_traffic / max(range_keywords, 1)
                })
            
            funnel_df = pd.DataFrame(funnel_data)
            
            # Create funnel chart
            fig = go.Figure(go.Funnel(
                y=funnel_df['range'],
                x=funnel_df['traffic'],
                textinfo="value+percent initial",
                marker=dict(
                    color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"]
                )
            ))
            
            fig.update_layout(
                title="Traffic Distribution by Position Ranges",
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating traffic funnel chart: {str(e)}")
            return go.Figure()

    def create_serp_feature_analysis(
        self,
        serp_data: pd.DataFrame,
        competitor_comparison: bool = True
    ) -> go.Figure:
        """
        Create SERP feature presence analysis.
        
        Args:
            serp_data: DataFrame with SERP feature data
            competitor_comparison: Whether to compare across competitors
            
        Returns:
            Plotly figure with SERP feature analysis
        """
        try:
            # Parse SERP features
            feature_data = self._parse_serp_features(serp_data)
            
            if competitor_comparison and 'competitor' in serp_data.columns:
                fig = self._create_serp_competitor_comparison(feature_data)
            else:
                fig = self._create_serp_feature_distribution(feature_data)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating SERP feature analysis: {str(e)}")
            return go.Figure()

    def _create_position_line_chart(
        self, 
        data: pd.DataFrame, 
        keywords: List[str]
    ) -> go.Figure:
        """Create line chart for position tracking."""
        fig = go.Figure()
        
        for keyword in keywords:
            keyword_data = data[data['Keyword'] == keyword].sort_values('date')
            
            if not keyword_data.empty:
                fig.add_trace(go.Scatter(
                    x=keyword_data['date'],
                    y=keyword_data['Position'],
                    mode='lines+markers',
                    name=keyword,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        
        return fig

    def _create_competitive_scatter(
        self, 
        combined_df: pd.DataFrame, 
        metric: str
    ) -> go.Figure:
        """Create competitive scatter plot."""
        fig = px.scatter(
            combined_df,
            x='Position',
            y=metric,
            color='competitor',
            size='Search Volume' if 'Search Volume' in combined_df.columns else None,
            hover_data=['Keyword'],
            color_discrete_map=self.competitor_colors,
            title=f"Competitive Analysis: Position vs {metric}"
        )
        
        return fig

    def _parse_serp_features(self, serp_data: pd.DataFrame) -> pd.DataFrame:
        """Parse SERP features from string format."""
        feature_rows = []
        
        for _, row in serp_data.iterrows():
            features_str = row.get('SERP Features by Keyword', '')
            
            if pd.notna(features_str) and features_str:
                features = [f.strip() for f in str(features_str).split(',')]
                
                for feature in features:
                    feature_rows.append({
                        'Keyword': row.get('Keyword', ''),
                        'competitor': row.get('competitor', 'unknown'),
                        'Feature': feature,
                        'Position': row.get('Position', 100),
                        'Traffic': row.get('Traffic', 0)
                    })
        
        return pd.DataFrame(feature_rows)


class DashboardCreator:
    """
    Advanced dashboard creation for SEO competitive intelligence.
    
    Creates interactive dashboards combining multiple visualizations
    for comprehensive SEO analysis and reporting.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.chart_generator = ChartGenerator(logger)

    def create_executive_dashboard(
        self,
        position_data: pd.DataFrame,
        traffic_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create executive dashboard with key SEO metrics.
        
        Args:
            position_data: Position tracking data
            traffic_data: Traffic performance data
            competitor_data: Competitor analysis data
            save_path: Optional path to save dashboard
            
        Returns:
            HTML string of the dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Position Trends', 'Traffic Distribution',
                    'Competitive Landscape', 'SERP Features',
                    'Keyword Performance', 'Growth Opportunities'
                ],
                specs=[
                    [{"secondary_y": False}, {"type": "pie"}],
                    [{"colspan": 2}, None],
                    [{"secondary_y": True}, {"type": "bar"}]
                ]
            )
            
            # Add position trends
            self._add_position_trends_to_subplot(fig, position_data, 1, 1)
            
            # Add traffic distribution
            self._add_traffic_distribution_to_subplot(fig, traffic_data, 1, 2)
            
            # Add competitive landscape
            self._add_competitive_landscape_to_subplot(fig, competitor_data, 2, 1)
            
            # Add keyword performance
            self._add_keyword_performance_to_subplot(fig, position_data, traffic_data, 3, 1)
            
            # Add growth opportunities
            self._add_growth_opportunities_to_subplot(fig, position_data, competitor_data, 3, 2)
            
            # Update layout
            fig.update_layout(
                title="SEO Competitive Intelligence Dashboard",
                height=1200,
                showlegend=True
            )
            
            # Generate HTML
            dashboard_html = self._generate_dashboard_html(fig)
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(dashboard_html)
                self.logger.info(f"Executive dashboard saved to {save_path}")
            
            return dashboard_html
            
        except Exception as e:
            self.logger.error(f"Error creating executive dashboard: {str(e)}")
            return ""

    def _generate_dashboard_html(self, fig: go.Figure) -> str:
        """Generate complete HTML for dashboard."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SEO Competitive Intelligence Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metrics-summary {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
                .metric-card {{ 
                    background: #f8f9fa; 
                    padding: 15px; 
                    border-radius: 5px; 
                    text-align: center;
                    min-width: 150px;
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007db8; }}
                .metric-label {{ color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SEO Competitive Intelligence Dashboard</h1>
                <p>Generated on {timestamp}</p>
            </div>
            
            <div class="metrics-summary">
                <div class="metric-card">
                    <div class="metric-value">1,234</div>
                    <div class="metric-label">Total Keywords</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">56%</div>
                    <div class="metric-label">Top 10 Positions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">+12%</div>
                    <div class="metric-label">Traffic Growth</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">8.5/10</div>
                    <div class="metric-label">Competitive Score</div>
                </div>
            </div>
            
            <div id="dashboard-chart">
                {plot_div}
            </div>
        </body>
        </html>
        """.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            plot_div=pyo.plot(fig, output_type='div', include_plotlyjs=False)
        )
        
        return html_template


class ReportVisualizer:
    """
    Specialized report visualization for SEO competitive intelligence.
    
    Creates publication-ready charts and visualizations for
    reports, presentations, and stakeholder communications.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Report styling
        self.report_style = {
            'figure_size': (10, 6),
            'color_palette': ['#E31837', '#0096D6', '#007DB8', '#00BCF2', '#96CEB4'],
            'background_color': 'white',
            'grid_color': '#EEEEEE',
            'text_color': '#333333'
        }

    def create_monthly_report_charts(
        self,
        monthly_data: Dict[str, pd.DataFrame],
        save_directory: str
    ) -> Dict[str, str]:
        """
        Create comprehensive monthly report visualizations.
        
        Args:
            monthly_data: Dictionary of monthly performance data
            save_directory: Directory to save charts
            
        Returns:
            Dictionary of chart file paths
        """
        try:
            chart_paths = {}
            
            # Position performance summary
            position_chart = self._create_position_summary_chart(
                monthly_data.get('positions', pd.DataFrame())
            )
            position_path = f"{save_directory}/position_performance.png"
            position_chart.write_image(position_path, width=1200, height=800, scale=2)
            chart_paths['position_performance'] = position_path
            
            # Traffic growth analysis
            traffic_chart = self._create_traffic_growth_chart(
                monthly_data.get('traffic', pd.DataFrame())
            )
            traffic_path = f"{save_directory}/traffic_growth.png"
            traffic_chart.write_image(traffic_path, width=1200, height=800, scale=2)
            chart_paths['traffic_growth'] = traffic_path
            
            # Competitive positioning
            competitive_chart = self._create_competitive_positioning_chart(
                monthly_data.get('competitors', {})
            )
            competitive_path = f"{save_directory}/competitive_positioning.png"
            competitive_chart.write_image(competitive_path, width=1200, height=800, scale=2)
            chart_paths['competitive_positioning'] = competitive_path
            
            # Opportunity analysis
            opportunity_chart = self._create_opportunity_analysis_chart(
                monthly_data.get('opportunities', pd.DataFrame())
            )
            opportunity_path = f"{save_directory}/opportunity_analysis.png"
            opportunity_chart.write_image(opportunity_path, width=1200, height=800, scale=2)
            chart_paths['opportunity_analysis'] = opportunity_path
            
            self.logger.info(f"Created {len(chart_paths)} monthly report charts")
            return chart_paths
            
        except Exception as e:
            self.logger.error(f"Error creating monthly report charts: {str(e)}")
            return {}

    def _create_position_summary_chart(self, position_data: pd.DataFrame) -> go.Figure:
        """Create position performance summary chart."""
        if position_data.empty:
            return go.Figure()
        
        # Calculate position distribution
        position_ranges = {
            'Top 3': (position_data['Position'] <= 3).sum(),
            'Top 10': ((position_data['Position'] > 3) & (position_data['Position'] <= 10)).sum(),
            'Top 20': ((position_data['Position'] > 10) & (position_data['Position'] <= 20)).sum(),
            'Beyond 20': (position_data['Position'] > 20).sum()
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(position_ranges.keys()),
                y=list(position_ranges.values()),
                marker_color=self.report_style['color_palette'][:4]
            )
        ])
        
        fig.update_layout(
            title="Keyword Position Distribution",
            xaxis_title="Position Range",
            yaxis_title="Number of Keywords",
            plot_bgcolor=self.report_style['background_color']
        )
        
        return fig
