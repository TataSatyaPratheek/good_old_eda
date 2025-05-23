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
        analysis_results: Dict[str, Any] = None,
        export_path: Union[str, Path] = None,
        format: str = 'html',
        include_charts: bool = True,
        # Parameters from paste file for backward compatibility
        data: Dict[str, Any] = None,
        output_path: str = None
    ) -> bool:
        """
        Export comprehensive executive report.
        
        Args:
            analysis_results: Analysis results dictionary (preferred parameter)
            export_path: Output file path (preferred parameter)
            format: Export format ('html', 'pdf', 'docx')
            include_charts: Whether to include charts
            data: Analysis data (backward compatibility)
            output_path: Output path (backward compatibility)
            
        Returns:
            True if export successful
        """
        try:
            # Handle backward compatibility
            if data is not None and analysis_results is None:
                analysis_results = data
            if output_path is not None and export_path is None:
                export_path = output_path
            
            if analysis_results is None or export_path is None:
                raise ValueError("Missing required parameters: analysis_results and export_path")
            
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            if format == 'html':
                return self._export_html_report(analysis_results, export_path, include_charts)
            elif format == 'pdf':
                return self._export_pdf_report(analysis_results, export_path, include_charts)
            elif format == 'docx':
                return self._export_docx_report(analysis_results, export_path, include_charts)
            elif format == 'json':
                # JSON fallback from paste file
                with open(export_path, 'w') as f:
                    json.dump(analysis_results, f, indent=2, default=str)
                self.logger.info(f"Exported executive report to {export_path}")
                return True
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

    # Additional methods from paste file
    def export_analysis_report(
        self,
        data: Dict[str, Any],
        output_path: str
    ) -> bool:
        """Export analysis report (from paste file)"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Exported analysis report to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export analysis report: {str(e)}")
            return False

    def export_data_quality_report(
        self,
        quality_data: pd.DataFrame,
        output_path: str
    ) -> bool:
        """Export data quality report (from paste file)"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            quality_data.to_excel(output_path, index=False)
            self.logger.info(f"Exported data quality report to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export data quality report: {str(e)}")
            return False

    def export_strategic_recommendations_report(
        self,
        recommendations: Dict[str, Any],
        output_path: str
    ) -> bool:
        """Export strategic recommendations (from paste file)"""
        return self.export_analysis_report(recommendations, output_path)

    def export_competitive_intelligence_briefing(
        self,
        intelligence: Dict[str, Any],
        output_path: str
    ) -> bool:
        """Export competitive intelligence briefing (from paste file)"""
        return self.export_executive_report(data=intelligence, output_path=output_path)

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

    def _export_pdf_report(
        self,
        analysis_results: Dict[str, Any],
        export_path: Path,
        include_charts: bool
    ) -> bool:
        """Export PDF format report."""
        try:
            # This would implement PDF export functionality
            # For now, fallback to HTML
            html_path = export_path.with_suffix('.html')
            return self._export_html_report(analysis_results, html_path, include_charts)
        except Exception as e:
            self.logger.error(f"Error exporting PDF report: {str(e)}")
            return False

    def _export_docx_report(
        self,
        analysis_results: Dict[str, Any],
        export_path: Path,
        include_charts: bool
    ) -> bool:
        """Export DOCX format report."""
        try:
            # This would implement DOCX export functionality
            # For now, fallback to HTML
            html_path = export_path.with_suffix('.html')
            return self._export_html_report(analysis_results, html_path, include_charts)
        except Exception as e:
            self.logger.error(f"Error exporting DOCX report: {str(e)}")
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

        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>SEO Intelligence Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric {{ margin: 10px 0; padding: 10px; background-color: #ecf0f1; border-radius: 5px; }}
        .summary {{ background-color: #e8f4fd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .recommendations {{ background-color: #d5f4e6; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .chart-placeholder {{ background-color: #f8f9fa; padding: 40px; text-align: center; border: 2px dashed #dee2e6; margin: 20px 0; }}
        .data-section {{ margin: 20px 0; }}
        .metric-value {{ font-weight: bold; color: #3498db; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SEO Competitive Intelligence Report</h1>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <p>This report provides comprehensive analysis of SEO performance and competitive positioning for the analyzed domain. 
            Key findings include position trends, traffic analysis, and competitive landscape insights.</p>
        </div>

        <div class="data-section">
            <h2>Position Analysis</h2>
            <div class="metric">
                <strong>Average Position:</strong> <span class="metric-value">{position_metrics.get('avg_position', 'N/A')}</span>
            </div>
            <div class="metric">
                <strong>Top 10 Ratio:</strong> <span class="metric-value">{position_metrics.get('top_10_ratio', 'N/A')}</span>
            </div>
            <div class="metric">
                <strong>Position Volatility:</strong> <span class="metric-value">{position_metrics.get('position_volatility', 'N/A')}</span>
            </div>
        </div>

        <div class="data-section">
            <h2>Traffic Analysis</h2>
            <div class="metric">
                <strong>Total Traffic:</strong> <span class="metric-value">{traffic_metrics.get('total_traffic', 'N/A')}</span>
            </div>
            <div class="metric">
                <strong>Traffic Growth:</strong> <span class="metric-value">{traffic_metrics.get('traffic_growth', 'N/A')}</span>
            </div>
        </div>

        <div class="data-section">
            <h2>Competitive Analysis</h2>
            <div class="metric">
                <strong>Competitive Position:</strong> <span class="metric-value">{competitive_metrics.get('competitive_position', 'N/A')}</span>
            </div>
            <div class="metric">
                <strong>Market Share:</strong> <span class="metric-value">{competitive_metrics.get('market_share', 'N/A')}</span>
            </div>
        </div>

        <div class="recommendations">
            <h2>Strategic Recommendations</h2>
            <ul>
                <li>Focus on keywords ranking in positions 4-10 for quick wins</li>
                <li>Develop content targeting high-volume, low-competition keywords</li>
                <li>Monitor competitor activities and respond to strategic moves</li>
            </ul>
        </div>

        {"<div class='chart-placeholder'><h3>Interactive Analysis Dashboard</h3><p>Charts would be embedded here when include_charts=True</p></div>" if include_charts else ""}
        
        <div class="data-section">
            <h2>Raw Data</h2>
            <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 12px;">
{json.dumps(analysis_results, indent=2, default=str)}
            </pre>
        </div>
    </div>
</body>
</html>"""

        return html_template

    def _generate_html_report(self, data: Dict[str, Any], include_charts: bool) -> str:
        """Generate simple HTML report (from paste file for compatibility)"""
        html = f"""<html>
<head>
    <title>SEO Intelligence Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metric {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>SEO Intelligence Report</h1>
    <div class="content">
        {json.dumps(data, indent=2, default=str).replace('\n', '<br>')}
    </div>
</body>
</html>"""
        return html

    def _initialize_export_templates(self) -> Dict[str, str]:
        """Initialize export templates."""
        return {
            'executive': 'executive_template.html',
            'competitive': 'competitive_template.html',
            'keyword': 'keyword_template.html'
        }

    def _generate_competitive_report_content(
        self,
        competitive_data: Dict[str, pd.DataFrame],
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate competitive report content."""
        return {
            'competitive_data': competitive_data,
            'analysis_results': analysis_results,
            'timestamp': datetime.now().isoformat()
        }

    def _create_competitive_html_report(self, report_content: Dict[str, Any]) -> str:
        """Create competitive analysis HTML report."""
        return self._generate_html_report_content(report_content, True)

    def _analyze_keyword_performance(
        self,
        keyword_data: pd.DataFrame,
        performance_metrics: Dict[str, Any],
        top_n_keywords: int
    ) -> Dict[str, Any]:
        """Analyze keyword performance for reporting."""
        try:
            # Get top performing keywords
            if 'Traffic (%)' in keyword_data.columns:
                top_keywords = keyword_data.nlargest(top_n_keywords, 'Traffic (%)')
            else:
                top_keywords = keyword_data.head(top_n_keywords)

            return {
                'top_keywords': top_keywords.to_dict('records'),
                'performance_metrics': performance_metrics,
                'total_keywords': len(keyword_data)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing keyword performance: {str(e)}")
            return {}

    def _create_keyword_performance_html(self, keyword_analysis: Dict[str, Any]) -> str:
        """Create keyword performance HTML report."""
        return self._generate_html_report_content(keyword_analysis, True)


class DataExporter:
    """Export data in various formats (from paste file)"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def export_analysis_dataset(
        self,
        datasets: Dict[str, pd.DataFrame],
        output_path: str
    ) -> bool:
        """Export multiple datasets to Excel"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, df in datasets.items():
                    # Truncate sheet name if too long
                    sheet_name = sheet_name[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.logger.info(f"Exported {len(datasets)} datasets to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export datasets: {str(e)}")
            return False
    
    def export_with_metadata(
        self,
        data: pd.DataFrame,
        metadata: Dict[str, Any],
        export_path: str
    ) -> bool:
        """Export data with metadata"""
        try:
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                # Write main data
                data.to_excel(writer, sheet_name='Data', index=False)
                
                # Write metadata
                metadata_df = pd.DataFrame([metadata])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            self.logger.info(f"Exported data with metadata to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export with metadata: {str(e)}")
            return False

    def export_to_csv(
        self,
        data: pd.DataFrame,
        output_path: str,
        encoding: str = 'utf-8'
    ) -> bool:
        """Export DataFrame to CSV"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(output_path, index=False, encoding=encoding)
            self.logger.info(f"Exported CSV to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {str(e)}")
            return False

    def export_to_json(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        output_path: str,
        orient: str = 'records'
    ) -> bool:
        """Export data to JSON"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(data, pd.DataFrame):
                data.to_json(output_path, orient=orient, indent=2)
            else:
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Exported JSON to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export JSON: {str(e)}")
            return False

    def export_compressed_dataset(
        self,
        datasets: Dict[str, pd.DataFrame],
        output_path: str,
        compression: str = 'zip'
    ) -> bool:
        """Export datasets as compressed archive"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if compression == 'zip':
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for name, df in datasets.items():
                        # Create CSV in memory
                        csv_buffer = BytesIO()
                        df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        # Add to zip
                        zipf.writestr(f"{name}.csv", csv_buffer.getvalue())
            
            self.logger.info(f"Exported compressed datasets to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export compressed dataset: {str(e)}")
            return False
