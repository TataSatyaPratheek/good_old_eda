"""
Simple Report Generator
Clean, readable reports without complexity
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class ReportGenerator:
    """Generate simple, clean reports"""
    
    def __init__(self):
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
    
    def generate_all_reports(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate all reports"""
        
        print("📝 Generating reports...")
        
        reports = {
            'html': self._generate_html_report(analysis_results),
            'json': self._generate_json_report(analysis_results),
            'excel': self._generate_excel_report(analysis_results)
        }
        
        print("✅ Reports generated successfully!")
        return reports
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report"""
        
        summary = results.get('summary', {})
        competitive = results.get('competitive', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SEO Competitive Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .competitor {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SEO Competitive Analysis Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Analysis Period: May 19-21, 2025</p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                {self._create_summary_html(summary)}
            </div>
            
            <div class="section">
                <h2>Competitive Analysis</h2>
                {self._create_competitive_html(competitive)}
            </div>
        </body>
        </html>
        """
        
        report_path = self.reports_dir / "seo_analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        return str(report_path)
    
    def _create_summary_html(self, summary: Dict[str, Any]) -> str:
        """Create summary section HTML"""
        
        html = ""
        for company, metrics in summary.items():
            if company == 'gap_keywords':
                continue
                
            html += f"""
            <div class="metric">
                <h3>{company.title()}</h3>
                <p><strong>Total Keywords:</strong> {metrics.get('total_keywords', 0):,}</p>
                <p><strong>Average Position:</strong> {metrics.get('avg_position', 0):.1f}</p>
                <p><strong>Total Traffic:</strong> {metrics.get('total_traffic', 0):.1f}%</p>
                <p><strong>Top 10 Rankings:</strong> {metrics.get('top_10_count', 0):,}</p>
            </div>
            """
        
        return html
    
    def _create_competitive_html(self, competitive: Dict[str, Any]) -> str:
        """Create competitive analysis HTML"""
        
        market_share = competitive.get('market_share', {})
        
        html = "<h3>Market Share</h3>"
        for company, share in market_share.items():
            html += f"""
            <div class="competitor">
                <strong>{company.title()}:</strong> {share:.1f}%
            </div>
            """
        
        return html
    
    def _generate_json_report(self, results: Dict[str, Any]) -> str:
        """Generate JSON report"""
        
        report_path = self.reports_dir / "seo_analysis_data.json"
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(report_path)
    
    def _generate_excel_report(self, results: Dict[str, Any]) -> str:
        """Generate Excel report"""
        
        # Simple Excel export would go here
        # For now, just create a placeholder
        report_path = self.reports_dir / "seo_analysis_data.xlsx"
        
        # Could use pandas.ExcelWriter for real implementation
        with open(report_path, 'w') as f:
            f.write("Excel report placeholder")
        
        return str(report_path)
