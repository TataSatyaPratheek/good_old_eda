"""
Export Utilities for SEO Competitive Intelligence
Advanced export capabilities for reports, data, and visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime
import json
import base64
from io import BytesIO
import zipfile
from dataclasses import dataclass, asdict

@dataclass
class ExportConfiguration:
    """Export configuration settings"""
    format: str
    compression: bool
    include_metadata: bool
    date_format: str
    decimal_places: int
    include_index: bool
    custom_headers: Dict[str, str]

class ReportExporter:
    """
    Advanced report export for SEO competitive intelligence.
    
    Handles comprehensive report generation with multiple formats,
    embedded visualizations, and executive summaries.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.export_templates = self._initialize_export_templates()

    def export_executive_report(
        self,
        analysis_results: Dict[str, Any],
        export_path: Union[str, Path],
        format: str = 'html',
        include_charts: bool = True
    ) -> bool:
        """
        Export comprehensive executive report.
        
        Args:
            analysis_results: Analysis results dictionary
            export_path: Output file path
            format: Export format ('html', 'pdf', 'docx')
            include_charts: Whether to include charts
            
        Returns:
            True if export successful
        """
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'html':
                return self._export_html_report(analysis_results, export_path, include_charts)
            elif format == 'pdf':
                return self._export_pdf_report(analysis_results, export_path, include_charts)
            elif format == 'docx':
                return self._export_docx_report(analysis_results, export_path, include_charts)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting executive report: {str(e)}")
            return False

    def export_competitive_analysis_report(
        self,
        competitive_data: Dict[str, pd.DataFrame],
        analysis_results: Dict[str, Any],
        export_path: Union[str, Path]
    ) -> bool:
        """
        Export competitive analysis report with detailed insights.
        
        Args:
            competitive_data: Competitive data by competitor
            analysis_results: Analysis results
            export_path: Output file path
            
        Returns:
            True if export successful
        """
        try:
            export_path = Path(export_path)
            
            # Generate comprehensive competitive report
            report_content = self._generate_competitive_report_content(
                competitive_data, analysis_results
            )
            
            # Export as HTML with embedded charts
            html_content = self._create_competitive_html_report(report_content)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Competitive analysis report exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting competitive analysis report: {str(e)}")
            return False

    def export_keyword_performance_report(
        self,
        keyword_data: pd.DataFrame,
        performance_metrics: Dict[str, Any],
        export_path: Union[str, Path],
        top_n_keywords: int = 50
    ) -> bool:
        """
        Export keyword performance report.
        
        Args:
            keyword_data: Keyword performance data
            performance_metrics: Performance metrics
            export_path: Output file path
            top_n_keywords: Number of top keywords to include
            
        Returns:
            True if export successful
        """
        try:
            export_path = Path(export_path)
            
            # Prepare keyword analysis
            keyword_analysis = self._analyze_keyword_performance(
                keyword_data, performance_metrics, top_n_keywords
            )
            
            # Generate report content
            report_html = self._create_keyword_performance_html(keyword_analysis)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            
            self.logger.info(f"Keyword performance report exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting keyword performance report: {str(e)}")
            return False

    def _export_html_report(
        self,
        analysis_results: Dict[str, Any],
        export_path: Path,
        include_charts: bool
    ) -> bool:
        """Export HTML format report."""
        try:
            # Generate HTML content
            html_content = self._generate_html_report_content(analysis_results, include_charts)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting HTML report: {str(e)}")
            return False

    def _generate_html_report_content(
        self,
        analysis_results: Dict[str, Any],
        include_charts: bool
    ) -> str:
        """Generate HTML report content."""
        
        # Extract key metrics
        position_metrics = analysis_results.get('position_analysis', {})
        traffic_metrics = analysis_results.get('traffic_analysis', {})
        competitive_metrics = analysis_results.get('competitive_analysis', {})
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SEO Competitive Intelligence Report</title>
            <style>
                {self._get_report_css()}
            </style>
            {self._get_chart_scripts() if include_charts else ''}
        </head>
        <body>
            <div class="container">
                <header class="report-header">
                    <h1>SEO Competitive Intelligence Report</h1>
                    <p class="report-date">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </header>
                
                <section class="executive-summary">
                    <h2>Executive Summary</h2>
                    {self._generate_executive_summary(analysis_results)}
                </section>
                
                <section class="key-metrics">
                    <h2>Key Performance Metrics</h2>
                    {self._generate_key_metrics_section(position_metrics, traffic_metrics)}
                </section>
                
                <section class="competitive-landscape">
                    <h2>Competitive Landscape</h2>
                    {self._generate_competitive_section(competitive_metrics)}
                </section>
                
                <section class="recommendations">
                    <h2>Strategic Recommendations</h2>
                    {self._generate_recommendations_section(analysis_results)}
                </section>
                
                {self._generate_charts_section(analysis_results) if include_charts else ''}
                
                <footer class="report-footer">
                    <p>Report generated by SEO Competitive Intelligence Platform</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html_template

    def _get_report_css(self) -> str:
        """Get CSS styles for HTML reports."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        .report-header {
            text-align: center;
            border-bottom: 3px solid #007db8;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .report-header h1 {
            color: #007db8;
            margin: 0;
            font-size: 2.5em;
        }
        
        .report-date {
            color: #666;
            font-style: italic;
        }
        
        section {
            margin-bottom: 40px;
        }
        
        h2 {
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        
        .metric-card {
            display: inline-block;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 10px;
            min-width: 200px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #007db8;
        }
        
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        
        .recommendation {
            background: #e8f4f8;
            border-left: 4px solid #007db8;
            padding: 15px;
            margin: 10px 0;
        }
        
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        
        .report-footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
        }
        """

    def _initialize_export_templates(self) -> Dict[str, str]:
        """Initialize export templates."""
        return {
            'executive_summary': """
            <div class="executive-summary-content">
                <p>This report provides comprehensive analysis of SEO performance and competitive positioning 
                for the analyzed domain. Key findings include position trends, traffic analysis, and 
                competitive landscape insights.</p>
            </div>
            """,
            'recommendations': """
            <div class="recommendations-list">
                <div class="recommendation">
                    <h4>Immediate Opportunities</h4>
                    <p>Focus on keywords ranking in positions 4-10 for quick wins</p>
                </div>
                <div class="recommendation">
                    <h4>Content Strategy</h4>
                    <p>Develop content targeting high-volume, low-competition keywords</p>
                </div>
                <div class="recommendation">
                    <h4>Competitive Response</h4>
                    <p>Monitor competitor activities and respond to strategic moves</p>
                </div>
            </div>
            """
        }


class DataExporter:
    """
    Advanced data export utilities for SEO competitive intelligence.
    
    Provides flexible data export capabilities with multiple formats,
    compression, and metadata inclusion.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def export_analysis_dataset(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        export_path: Union[str, Path],
        config: Optional[ExportConfiguration] = None
    ) -> bool:
        """
        Export analysis dataset with comprehensive configuration options.
        
        Args:
            data: Data to export (DataFrame or dict of DataFrames)
            export_path: Output file path
            config: Export configuration
            
        Returns:
            True if export successful
        """
        try:
            if config is None:
                config = ExportConfiguration(
                    format='csv',
                    compression=False,
                    include_metadata=True,
                    date_format='%Y-%m-%d',
                    decimal_places=3,
                    include_index=False,
                    custom_headers={}
                )
            
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(data, pd.DataFrame):
                return self._export_single_dataframe(data, export_path, config)
            elif isinstance(data, dict):
                return self._export_multiple_dataframes(data, export_path, config)
            else:
                raise ValueError("Data must be DataFrame or dict of DataFrames")
                
        except Exception as e:
            self.logger.error(f"Error exporting analysis dataset: {str(e)}")
            return False

    def export_with_metadata(
        self,
        data: pd.DataFrame,
        metadata: Dict[str, Any],
        export_path: Union[str, Path],
        format: str = 'excel'
    ) -> bool:
        """
        Export data with comprehensive metadata.
        
        Args:
            data: DataFrame to export
            metadata: Metadata dictionary
            export_path: Output file path
            format: Export format
            
        Returns:
            True if export successful
        """
        try:
            export_path = Path(export_path)
            
            if format == 'excel':
                with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                    # Write main data
                    data.to_excel(writer, sheet_name='Data', index=False)
                    
                    # Write metadata
                    metadata_df = pd.DataFrame(list(metadata.items()), columns=['Key', 'Value'])
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                    
                    # Write data dictionary if available
                    if 'data_dictionary' in metadata:
                        dict_df = pd.DataFrame(metadata['data_dictionary'])
                        dict_df.to_excel(writer, sheet_name='Data_Dictionary', index=False)
            
            elif format == 'json':
                export_data = {
                    'data': data.to_dict('records'),
                    'metadata': metadata,
                    'export_timestamp': datetime.now().isoformat()
                }
                
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            else:
                raise ValueError(f"Unsupported format for metadata export: {format}")
            
            self.logger.info(f"Data with metadata exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data with metadata: {str(e)}")
            return False

    def create_data_package(
        self,
        datasets: Dict[str, pd.DataFrame],
        package_path: Union[str, Path],
        include_documentation: bool = True
    ) -> bool:
        """
        Create comprehensive data package with multiple datasets.
        
        Args:
            datasets: Dictionary of datasets to package
            package_path: Output package path (zip file)
            include_documentation: Whether to include documentation
            
        Returns:
            True if package created successfully
        """
        try:
            package_path = Path(package_path)
            package_path.parent.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Export each dataset
                for name, df in datasets.items():
                    # Create temporary CSV
                    csv_content = df.to_csv(index=False)
                    zipf.writestr(f"{name}.csv", csv_content)
                    
                    # Create data summary
                    summary = self._create_dataset_summary(df, name)
                    zipf.writestr(f"{name}_summary.json", json.dumps(summary, indent=2, default=str))
                
                # Include package metadata
                package_metadata = {
                    'created_by': 'SEO Competitive Intelligence Platform',
                    'creation_date': datetime.now().isoformat(),
                    'datasets': list(datasets.keys()),
                    'total_records': sum(len(df) for df in datasets.values()),
                    'package_version': '1.0'
                }
                zipf.writestr('package_metadata.json', json.dumps(package_metadata, indent=2))
                
                # Include documentation if requested
                if include_documentation:
                    documentation = self._generate_package_documentation(datasets)
                    zipf.writestr('README.md', documentation)
            
            self.logger.info(f"Data package created: {package_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating data package: {str(e)}")
            return False

    def _export_single_dataframe(
        self,
        df: pd.DataFrame,
        export_path: Path,
        config: ExportConfiguration
    ) -> bool:
        """Export single DataFrame with configuration."""
        try:
            # Apply configuration
            export_df = df.copy()
            
            # Apply custom headers
            if config.custom_headers:
                export_df = export_df.rename(columns=config.custom_headers)
            
            # Format numeric columns
            numeric_columns = export_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                export_df[col] = export_df[col].round(config.decimal_places)
            
            # Export based on format
            if config.format == 'csv':
                export_df.to_csv(
                    export_path,
                    index=config.include_index,
                    compression='gzip' if config.compression else None
                )
            elif config.format == 'excel':
                export_df.to_excel(export_path, index=config.include_index)
            elif config.format == 'json':
                export_df.to_json(
                    export_path,
                    orient='records',
                    date_format=config.date_format,
                    indent=2
                )
            elif config.format == 'parquet':
                export_df.to_parquet(
                    export_path,
                    compression='gzip' if config.compression else None
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting single DataFrame: {str(e)}")
            return False

    def _create_dataset_summary(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """Create summary for dataset."""
        return {
            'name': name,
            'records': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'creation_date': datetime.now().isoformat()
        }


class VisualizationExporter:
    """
    Advanced visualization export for SEO competitive intelligence.
    
    Handles export of charts, dashboards, and interactive visualizations
    in multiple formats with customization options.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def export_chart_collection(
        self,
        charts: Dict[str, Any],
        export_directory: Union[str, Path],
        formats: List[str] = None,
        resolution: int = 300
    ) -> Dict[str, bool]:
        """
        Export collection of charts in multiple formats.
        
        Args:
            charts: Dictionary of chart objects
            export_directory: Output directory
            formats: Export formats ('png', 'svg', 'pdf', 'html')
            resolution: Image resolution for raster formats
            
        Returns:
            Dictionary of export results by chart name
        """
        try:
            if formats is None:
                formats = ['png', 'html']
            
            export_dir = Path(export_directory)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            results = {}
            
            for chart_name, chart_obj in charts.items():
                chart_results = {}
                
                for format in formats:
                    try:
                        output_path = export_dir / f"{chart_name}.{format}"
                        
                        if hasattr(chart_obj, 'write_image') and format in ['png', 'jpg', 'svg', 'pdf']:
                            # Plotly chart
                            chart_obj.write_image(
                                output_path,
                                width=1200,
                                height=800,
                                scale=resolution/100
                            )
                        elif hasattr(chart_obj, 'write_html') and format == 'html':
                            # Plotly HTML
                            chart_obj.write_html(output_path)
                        elif hasattr(chart_obj, 'savefig'):
                            # Matplotlib chart
                            chart_obj.savefig(output_path, dpi=resolution, bbox_inches='tight')
                        else:
                            # Generic export attempt
                            self._generic_chart_export(chart_obj, output_path, format)
                        
                        chart_results[format] = True
                        
                    except Exception as e:
                        self.logger.error(f"Error exporting {chart_name} as {format}: {str(e)}")
                        chart_results[format] = False
                
                results[chart_name] = chart_results
            
            successful_exports = sum(
                sum(1 for success in chart_results.values() if success)
                for chart_results in results.values()
            )
            
            self.logger.info(f"Chart collection export completed: {successful_exports} successful exports")
            return results
            
        except Exception as e:
            self.logger.error(f"Error exporting chart collection: {str(e)}")
            return {}

    def export_interactive_dashboard(
        self,
        dashboard_components: Dict[str, Any],
        export_path: Union[str, Path],
        template: str = 'bootstrap'
    ) -> bool:
        """
        Export interactive dashboard as standalone HTML.
        
        Args:
            dashboard_components: Dashboard components (charts, tables, etc.)
            export_path: Output HTML file path
            template: Dashboard template style
            
        Returns:
            True if export successful
        """
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate dashboard HTML
            dashboard_html = self._create_interactive_dashboard_html(
                dashboard_components, template
            )
            
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            
            self.logger.info(f"Interactive dashboard exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting interactive dashboard: {str(e)}")
            return False

    def export_chart_as_base64(
        self,
        chart_obj: Any,
        format: str = 'png',
        width: int = 800,
        height: int = 600
    ) -> Optional[str]:
        """
        Export chart as base64 encoded string for embedding.
        
        Args:
            chart_obj: Chart object to export
            format: Image format
            width: Image width
            height: Image height
            
        Returns:
            Base64 encoded string or None if failed
        """
        try:
            if hasattr(chart_obj, 'to_image'):
                # Plotly chart
                img_bytes = chart_obj.to_image(
                    format=format,
                    width=width,
                    height=height,
                    scale=2
                )
            elif hasattr(chart_obj, 'savefig'):
                # Matplotlib chart
                buffer = BytesIO()
                chart_obj.savefig(buffer, format=format, dpi=150, bbox_inches='tight')
                buffer.seek(0)
                img_bytes = buffer.getvalue()
            else:
                return None
            
            # Encode as base64
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/{format};base64,{base64_string}"
            
        except Exception as e:
            self.logger.error(f"Error converting chart to base64: {str(e)}")
            return None

    def _generic_chart_export(
        self,
        chart_obj: Any,
        output_path: Path,
        format: str
    ):
        """Generic chart export for unknown chart types."""
        # Try common export methods
        export_methods = ['save', 'export', 'write', 'to_file']
        
        for method_name in export_methods:
            if hasattr(chart_obj, method_name):
                method = getattr(chart_obj, method_name)
                try:
                    method(str(output_path))
                    return
                except:
                    continue
        
        raise ValueError(f"Unable to export chart of type {type(chart_obj)}")

    def _create_interactive_dashboard_html(
        self,
        components: Dict[str, Any],
        template: str
    ) -> str:
        """Create interactive dashboard HTML."""
        
        # Convert charts to HTML/base64
        chart_html_elements = []
        
        for name, component in components.items():
            if hasattr(component, 'to_html'):
                # Plotly chart
                chart_html = component.to_html(
                    include_plotlyjs='cdn',
                    div_id=f"chart_{name}"
                )
                chart_html_elements.append(f'<div class="chart-container">{chart_html}</div>')
            elif hasattr(component, 'to_dict'):
                # DataFrame table
                table_html = component.to_html(classes='table table-striped')
                chart_html_elements.append(f'<div class="table-container">{table_html}</div>')
            else:
                # Generic component
                chart_html_elements.append(f'<div class="component-container">{str(component)}</div>')
        
        dashboard_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SEO Competitive Intelligence Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                .chart-container {{ margin: 20px 0; }}
                .table-container {{ margin: 20px 0; }}
                .dashboard-header {{ 
                    background: linear-gradient(135deg, #007db8, #0096d6);
                    color: white;
                    padding: 30px 0;
                    margin-bottom: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <div class="container">
                    <h1>SEO Competitive Intelligence Dashboard</h1>
                    <p class="lead">Interactive Analysis Dashboard</p>
                </div>
            </div>
            
            <div class="container">
                {"".join(chart_html_elements)}
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        return dashboard_template
