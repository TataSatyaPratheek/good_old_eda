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
from pathlib import Path
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

    def _create_position_heatmap(
        self,
        data: pd.DataFrame,
        keywords: List[str]
    ) -> go.Figure:
        """Create heatmap for position tracking."""
        try:
            # Pivot data for heatmap
            pivot_data = data.pivot_table(
                index='Keyword',
                columns='date',
                values='Position',
                aggfunc='mean'
            )

            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlGn_r',
                hoverongaps=False
            ))

            fig.update_layout(
                title="Position Heatmap",
                xaxis_title="Date",
                yaxis_title="Keywords"
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error creating position heatmap: {str(e)}")
            return go.Figure()

    def _create_position_area_chart(
        self,
        data: pd.DataFrame,
        keywords: List[str]
    ) -> go.Figure:
        """Create area chart for position tracking."""
        fig = go.Figure()

        for keyword in keywords:
            keyword_data = data[data['Keyword'] == keyword].sort_values('date')
            if not keyword_data.empty:
                fig.add_trace(go.Scatter(
                    x=keyword_data['date'],
                    y=keyword_data['Position'],
                    mode='lines',
                    name=keyword,
                    fill='tonexty' if keyword != keywords[0] else 'tozeroy',
                    line=dict(width=0)
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

    def _create_competitive_bubble(
        self,
        combined_df: pd.DataFrame,
        metric: str
    ) -> go.Figure:
        """Create competitive bubble chart."""
        fig = px.scatter(
            combined_df,
            x='Position',
            y=metric,
            size='Search Volume' if 'Search Volume' in combined_df.columns else 'Traffic',
            color='competitor',
            hover_data=['Keyword'],
            size_max=60,
            color_discrete_map=self.competitor_colors,
            title=f"Competitive Bubble Analysis: {metric}"
        )
        return fig

    def _create_competitive_radar(
        self,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> go.Figure:
        """Create competitive radar chart."""
        fig = go.Figure()

        # Define metrics for radar chart
        metrics = ['avg_position', 'total_traffic', 'keyword_count', 'top_10_ratio']

        for competitor, df in competitor_data.items():
            # Calculate metrics
            values = [
                df['Position'].mean() if 'Position' in df.columns else 0,
                df['Traffic'].sum() if 'Traffic' in df.columns else 0,
                len(df),
                (df['Position'] <= 10).mean() if 'Position' in df.columns else 0
            ]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=competitor
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max(competitor_data[comp]['Position']) if 'Position' in competitor_data[comp].columns else 100 for comp in competitor_data.keys()])]
                )
            ),
            showlegend=True,
            title="Competitive Radar Analysis"
        )

        return fig

    def _create_competitive_bar(
        self,
        combined_df: pd.DataFrame,
        metric: str
    ) -> go.Figure:
        """Create competitive bar chart."""
        competitor_metrics = combined_df.groupby('competitor')[metric].sum().reset_index()

        fig = px.bar(
            competitor_metrics,
            x='competitor',
            y=metric,
            color='competitor',
            color_discrete_map=self.competitor_colors,
            title=f"Competitive Comparison: {metric}"
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

    def _create_serp_competitor_comparison(self, feature_data: pd.DataFrame) -> go.Figure:
        """Create SERP feature competitor comparison."""
        feature_counts = feature_data.groupby(['Feature', 'competitor']).size().reset_index(name='count')

        fig = px.bar(
            feature_counts,
            x='Feature',
            y='count',
            color='competitor',
            color_discrete_map=self.competitor_colors,
            title="SERP Features by Competitor"
        )

        fig.update_xaxis(tickangle=45)
        return fig

    def _create_serp_feature_distribution(self, feature_data: pd.DataFrame) -> go.Figure:
        """Create SERP feature distribution chart."""
        feature_counts = feature_data['Feature'].value_counts().reset_index()
        feature_counts.columns = ['Feature', 'Count']

        fig = px.pie(
            feature_counts,
            values='Count',
            names='Feature',
            title="SERP Feature Distribution"
        )

        return fig


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

    def _add_position_trends_to_subplot(
        self,
        fig: go.Figure,
        position_data: pd.DataFrame,
        row: int,
        col: int
    ):
        """Add position trends to subplot."""
        try:
            if 'date' in position_data.columns and 'Position' in position_data.columns:
                avg_positions = position_data.groupby('date')['Position'].mean().reset_index()
                fig.add_trace(
                    go.Scatter(
                        x=avg_positions['date'],
                        y=avg_positions['Position'],
                        mode='lines+markers',
                        name='Avg Position'
                    ),
                    row=row, col=col
                )
        except Exception as e:
            self.logger.error(f"Error adding position trends: {str(e)}")

    def _add_traffic_distribution_to_subplot(
        self,
        fig: go.Figure,
        traffic_data: pd.DataFrame,
        row: int,
        col: int
    ):
        """Add traffic distribution to subplot."""
        try:
            if 'Position' in traffic_data.columns and 'Traffic' in traffic_data.columns:
                # Create position ranges
                position_ranges = ['1-3', '4-10', '11-20', '21-50', '51+']
                traffic_by_range = []

                for i, range_label in enumerate(position_ranges):
                    if i == 0:
                        mask = traffic_data['Position'].between(1, 3)
                    elif i == 1:
                        mask = traffic_data['Position'].between(4, 10)
                    elif i == 2:
                        mask = traffic_data['Position'].between(11, 20)
                    elif i == 3:
                        mask = traffic_data['Position'].between(21, 50)
                    else:
                        mask = traffic_data['Position'] > 50

                    traffic_sum = traffic_data[mask]['Traffic'].sum()
                    traffic_by_range.append(traffic_sum)

                fig.add_trace(
                    go.Pie(
                        labels=position_ranges,
                        values=traffic_by_range,
                        name="Traffic Distribution"
                    ),
                    row=row, col=col
                )
        except Exception as e:
            self.logger.error(f"Error adding traffic distribution: {str(e)}")

    def _add_competitive_landscape_to_subplot(
        self,
        fig: go.Figure,
        competitor_data: Dict[str, pd.DataFrame],
        row: int,
        col: int
    ):
        """Add competitive landscape to subplot."""
        try:
            competitor_metrics = []
            for competitor, df in competitor_data.items():
                avg_position = df['Position'].mean() if 'Position' in df.columns else 0
                total_traffic = df['Traffic'].sum() if 'Traffic' in df.columns else 0
                competitor_metrics.append({
                    'competitor': competitor,
                    'avg_position': avg_position,
                    'total_traffic': total_traffic
                })

            if competitor_metrics:
                comp_df = pd.DataFrame(competitor_metrics)
                fig.add_trace(
                    go.Scatter(
                        x=comp_df['avg_position'],
                        y=comp_df['total_traffic'],
                        mode='markers+text',
                        text=comp_df['competitor'],
                        textposition="top center",
                        marker=dict(size=15),
                        name="Competitors"
                    ),
                    row=row, col=col
                )
        except Exception as e:
            self.logger.error(f"Error adding competitive landscape: {str(e)}")

    def _add_keyword_performance_to_subplot(
        self,
        fig: go.Figure,
        position_data: pd.DataFrame,
        traffic_data: pd.DataFrame,
        row: int,
        col: int
    ):
        """Add keyword performance to subplot."""
        try:
            if 'Keyword' in position_data.columns and 'Traffic' in traffic_data.columns:
                # Get top keywords by traffic
                top_keywords = traffic_data.groupby('Keyword')['Traffic'].sum().nlargest(10)

                fig.add_trace(
                    go.Bar(
                        x=list(top_keywords.index),
                        y=list(top_keywords.values),
                        name="Top Keywords by Traffic"
                    ),
                    row=row, col=col
                )
        except Exception as e:
            self.logger.error(f"Error adding keyword performance: {str(e)}")

    def _add_growth_opportunities_to_subplot(
        self,
        fig: go.Figure,
        position_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        row: int,
        col: int
    ):
        """Add growth opportunities to subplot."""
        try:
            # Find keywords ranking 11-20 (opportunity keywords)
            if 'Position' in position_data.columns and 'Search Volume' in position_data.columns:
                opportunity_keywords = position_data[
                    (position_data['Position'] >= 11) & 
                    (position_data['Position'] <= 20)
                ].nlargest(10, 'Search Volume')

                if not opportunity_keywords.empty:
                    fig.add_trace(
                        go.Bar(
                            x=opportunity_keywords['Keyword'],
                            y=opportunity_keywords['Search Volume'],
                            name="Growth Opportunities"
                        ),
                        row=row, col=col
                    )
        except Exception as e:
            self.logger.error(f"Error adding growth opportunities: {str(e)}")

    def _generate_dashboard_html(self, fig: go.Figure) -> str:
        """Generate complete HTML for dashboard."""
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>SEO Competitive Intelligence Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .dashboard-container {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SEO Competitive Intelligence Dashboard</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="dashboard-container">
        {fig.to_html(include_plotlyjs=False, div_id="dashboard")}
    </div>
    
    <div class="footer">
        <p>© 2025 SEO Intelligence Platform</p>
    </div>
</body>
</html>"""
        return html_template


class VisualizationEngine:
    """
    Create various visualizations for SEO data (from paste file)
    Simple matplotlib/seaborn-based visualization engine for basic charting needs
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.style_config = {
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14
        }
        self._setup_style()
    
    def _setup_style(self):
        """Setup visualization style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        for key, value in self.style_config.items():
            plt.rcParams[key] = value
    
    def create_position_trend_charts(
        self,
        data: pd.DataFrame,
        output_path: str
    ) -> bool:
        """Create position trend visualizations"""
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            if 'date' in data.columns and 'Position' in data.columns:
                fig, ax = plt.subplots(figsize=self.style_config['figure.figsize'])
                
                # Plot average position over time
                avg_position = data.groupby('date')['Position'].mean()
                ax.plot(avg_position.index, avg_position.values, marker='o')
                ax.set_xlabel('Date')
                ax.set_ylabel('Average Position')
                ax.set_title('Average Position Trend Over Time')
                ax.invert_yaxis()  # Lower position is better
                
                plt.tight_layout()
                plt.savefig(f"{output_path}/position_trend.png")
                plt.close()
                
                self.logger.info(f"Created position trend chart at {output_path}")
                return True
            else:
                self.logger.warning("Missing required columns for position trend chart")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating position trend charts: {str(e)}")
            return False
    
    def create_traffic_comparison_charts(
        self,
        traffic_data: Dict[str, Any],
        output_path: str
    ) -> bool:
        """Create traffic comparison visualizations"""
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            # Placeholder implementation
            fig, ax = plt.subplots(figsize=self.style_config['figure.figsize'])
            ax.text(0.5, 0.5, 'Traffic Comparison Chart', ha='center', va='center')
            ax.set_title('Traffic Performance Comparison')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}/traffic_comparison.png")
            plt.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating traffic comparison charts: {str(e)}")
            return False
    
    def create_competitive_landscape_chart(
        self,
        competitive_data: Dict[str, Any],
        output_path: str
    ) -> bool:
        """Create competitive landscape visualization"""
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            # Placeholder implementation
            fig, ax = plt.subplots(figsize=self.style_config['figure.figsize'])
            ax.text(0.5, 0.5, 'Competitive Landscape', ha='center', va='center')
            ax.set_title('Competitive Landscape Analysis')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}/competitive_landscape.png")
            plt.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating competitive landscape chart: {str(e)}")
            return False
    
    def create_market_share_charts(
        self,
        market_data: Dict[str, Any],
        output_path: str
    ) -> bool:
        """Create market share visualizations"""
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            # Placeholder implementation
            fig, ax = plt.subplots(figsize=self.style_config['figure.figsize'])
            
            # Simple pie chart placeholder
            sizes = [30, 25, 20, 15, 10]
            labels = ['Company A', 'Company B', 'Company C', 'Company D', 'Others']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title('Market Share Distribution')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}/market_share.png")
            plt.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating market share charts: {str(e)}")
            return False
    
    def create_heatmap(
        self,
        data: pd.DataFrame,
        title: str,
        output_path: str
    ) -> bool:
        """Create heatmap visualization"""
        try:
            fig, ax = plt.subplots(figsize=self.style_config['figure.figsize'])
            
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            if not numeric_data.empty:
                sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title(title)
                
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                
                self.logger.info(f"Created heatmap at {output_path}")
                return True
            else:
                self.logger.warning("No numeric data for heatmap")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating heatmap: {str(e)}")
            return False
    
    def create_distribution_plot(
        self,
        data: pd.Series,
        title: str,
        output_path: str
    ) -> bool:
        """Create distribution plot"""
        try:
            fig, ax = plt.subplots(figsize=self.style_config['figure.figsize'])
            
            sns.histplot(data=data, kde=True, ax=ax)
            ax.set_title(title)
            ax.set_xlabel(data.name if hasattr(data, 'name') else 'Value')
            ax.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            self.logger.info(f"Created distribution plot at {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating distribution plot: {str(e)}")
            return False

    def create_scatter_plot(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        output_path: str,
        color_col: Optional[str] = None
    ) -> bool:
        """Create scatter plot visualization"""
        try:
            fig, ax = plt.subplots(figsize=self.style_config['figure.figsize'])
            
            if color_col and color_col in data.columns:
                scatter = ax.scatter(data[x_col], data[y_col], c=data[color_col], alpha=0.6)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(data[x_col], data[y_col], alpha=0.6)
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            self.logger.info(f"Created scatter plot at {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {str(e)}")
            return False

    def create_line_plot(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        output_path: str,
        group_col: Optional[str] = None
    ) -> bool:
        """Create line plot visualization"""
        try:
            fig, ax = plt.subplots(figsize=self.style_config['figure.figsize'])
            
            if group_col and group_col in data.columns:
                for name, group in data.groupby(group_col):
                    ax.plot(group[x_col], group[y_col], marker='o', label=name)
                ax.legend()
            else:
                ax.plot(data[x_col], data[y_col], marker='o')
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            self.logger.info(f"Created line plot at {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating line plot: {str(e)}")
            return False
