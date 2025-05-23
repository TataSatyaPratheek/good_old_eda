"""
Enhanced Report Generator with Visualizations (Brick 3)
Clean, readable reports with charts and visual analysis
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np

class ReportGenerator:
    """Generate comprehensive reports with visualizations"""
    
    def __init__(self):
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.reports_dir / "visuals").mkdir(exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_all_reports(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate all reports"""
        
        print("📝 Generating comprehensive reports...")
        
        reports = {
            'html': self._generate_html_report(analysis_results),
            'json': self._generate_json_report(analysis_results),
            'excel': self._generate_excel_report(analysis_results)
        }
        
        print("✅ Reports generated successfully!")
        return reports
    
    def generate_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization charts (Brick 3)"""
        
        print("📊 Generating visualizations...")
        
        viz_dir = self.reports_dir / "visuals"
        viz_dir.mkdir(exist_ok=True)
        
        charts = {}
        
        try:
            # Market share pie chart
            competitive = results.get('competitive', {})
            if 'market_share' in competitive:
                charts['market_share'] = self._create_market_share_chart(competitive['market_share'], viz_dir)
            
            # Position distribution chart
            keyword_analysis = results.get('keyword_analysis', {})
            if 'position_distribution' in keyword_analysis:
                charts['position_dist'] = self._create_position_distribution_chart(keyword_analysis['position_distribution'], viz_dir)
            
            # Traffic comparison bar chart
            if 'traffic_comparison' in competitive:
                charts['traffic_comparison'] = self._create_traffic_comparison_chart(competitive['traffic_comparison'], viz_dir)
            
            # Opportunity analysis scatter plot
            opportunity_analysis = results.get('opportunity_analysis', {})
            if 'high_value_opportunities' in opportunity_analysis:
                charts['opportunities'] = self._create_opportunities_chart(opportunity_analysis['high_value_opportunities'], viz_dir)
            
            print(f"📊 Generated {len(charts)} visualizations")
            
        except Exception as e:
            print(f"⚠️ Error generating some visualizations: {e}")
        
        return charts
    
    def _create_market_share_chart(self, market_share: Dict[str, float], viz_dir: Path) -> str:
        """Create market share pie chart"""
        
        plt.figure(figsize=(10, 8))
        
        companies = list(market_share.keys())
        shares = list(market_share.values())
        colors = ['#2E86C1', '#F39C12', '#27AE60']  # Blue, Orange, Green
        
        # Create pie chart
        wedges, texts, autotexts = plt.pie(shares, labels=companies, autopct='%1.1f%%', 
                                          colors=colors, startangle=90, explode=(0.05, 0, 0))
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        for text in texts:
            text.set_fontsize(14)
            text.set_fontweight('bold')
        
        plt.title('Market Share by Organic Traffic', fontsize=16, fontweight='bold', pad=20)
        
        # Add total traffic info
        total_traffic = sum(shares)
        plt.figtext(0.5, 0.02, f'Total Traffic Analyzed: {total_traffic:.1f}%', 
                   ha='center', fontsize=10, style='italic')
        
        chart_path = viz_dir / "market_share.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(chart_path)
    
    def _create_position_distribution_chart(self, position_dist: Dict[str, int], viz_dir: Path) -> str:
        """Create position distribution bar chart"""
        
        plt.figure(figsize=(12, 8))
        
        positions = list(position_dist.keys())
        counts = list(position_dist.values())
        
        # Create bars with gradient colors
        bars = plt.bar(positions, counts, color=['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Lenovo Keyword Position Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Position Range', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Keywords', fontsize=12, fontweight='bold')
        
        # Improve grid
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add percentage labels
        total_keywords = sum(counts)
        if total_keywords > 0:
            for i, (pos, count) in enumerate(zip(positions, counts)):
                percentage = (count / total_keywords) * 100
                plt.text(i, count + max(counts)*0.05, f'({percentage:.1f}%)', 
                        ha='center', va='bottom', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        chart_path = viz_dir / "position_distribution.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(chart_path)
    
    def _create_traffic_comparison_chart(self, traffic_comparison: Dict[str, float], viz_dir: Path) -> str:
        """Create traffic comparison bar chart"""
        
        plt.figure(figsize=(10, 6))
        
        companies = list(traffic_comparison.keys())
        traffic = list(traffic_comparison.values())
        colors = ['#3498DB', '#E74C3C', '#2ECC71']
        
        bars = plt.bar(companies, traffic, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Traffic Share Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Company', fontsize=12, fontweight='bold')
        plt.ylabel('Traffic Share (%)', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        chart_path = viz_dir / "traffic_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(chart_path)
    
    def _create_opportunities_chart(self, opportunities: list, viz_dir: Path) -> str:
        """Create opportunities scatter plot"""
        
        if not opportunities:
            return ""
        
        plt.figure(figsize=(12, 8))
        
        # Extract data
        volumes = [opp.get('Volume', 0) for opp in opportunities]
        difficulties = [opp.get('Keyword Difficulty', 0) for opp in opportunities]
        keywords = [opp.get('Keyword', '')[:20] + '...' if len(opp.get('Keyword', '')) > 20 else opp.get('Keyword', '') for opp in opportunities]
        
        # Create scatter plot
        scatter = plt.scatter(difficulties, volumes, s=100, alpha=0.6, c=range(len(opportunities)), cmap='viridis')
        
        # Add labels for top opportunities
        for i, (diff, vol, keyword) in enumerate(zip(difficulties[:10], volumes[:10], keywords[:10])):
            plt.annotate(keyword, (diff, vol), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.title('SEO Opportunities: Volume vs Difficulty', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Keyword Difficulty', fontsize=12, fontweight='bold')
        plt.ylabel('Search Volume', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add ideal quadrant annotation
        plt.axvline(x=30, color='red', linestyle='--', alpha=0.5)
        plt.text(15, max(volumes)*0.9, 'Low Difficulty\n(Easier to Rank)', ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        
        chart_path = viz_dir / "opportunities_scatter.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(chart_path)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate enhanced HTML report"""
        
        summary = results.get('summary', {})
        competitive = results.get('competitive', {})
        keyword_analysis = results.get('keyword_analysis', {})
        opportunity_analysis = results.get('opportunity_analysis', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced SEO Competitive Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f8f9fa; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ background: white; margin: 20px 0; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ background: #e8f5e8; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #28a745; }}
                .competitor {{ background: #fff3cd; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #ffc107; }}
                .insight {{ background: #d1ecf1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #17a2b8; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .highlight {{ background-color: #fff3cd; font-weight: bold; }}
                .chart-container {{ text-align: center; margin: 20px 0; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #dee2e6; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .stat-label {{ font-size: 14px; color: #6c757d; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Enhanced SEO Competitive Intelligence Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Analysis Period:</strong> May 19-21, 2025</p>
                <p><strong>Features:</strong> Core Analysis • Competitive Comparison • Opportunities • Visualizations</p>
            </div>
            
            {self._create_executive_summary_section(summary, competitive)}
            {self._create_competitive_analysis_section(competitive)}
            {self._create_keyword_analysis_section(keyword_analysis)}
            {self._create_opportunity_section(opportunity_analysis)}
            {self._create_recommendations_section(results)}
        </body>
        </html>
        """
        
        report_path = self.reports_dir / "enhanced_seo_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(report_path)
    
    def _create_executive_summary_section(self, summary: Dict[str, Any], competitive: Dict[str, Any]) -> str:
        """Create executive summary section"""
        
        lenovo_data = summary.get('lenovo', {})
        market_share = competitive.get('market_share', {})
        
        html = f"""
        <div class="section">
            <h2>📈 Executive Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{lenovo_data.get('total_keywords', 0):,}</div>
                    <div class="stat-label">Total Keywords</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{lenovo_data.get('avg_position', 0):.1f}</div>
                    <div class="stat-label">Average Position</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{lenovo_data.get('total_traffic', 0):.1f}%</div>
                    <div class="stat-label">Traffic Share</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{market_share.get('lenovo', 0):.1f}%</div>
                    <div class="stat-label">Market Share</div>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _create_competitive_analysis_section(self, competitive: Dict[str, Any]) -> str:
        """Create competitive analysis section"""
        
        market_share = competitive.get('market_share', {})
        keyword_overlap = competitive.get('keyword_overlap', {})
        
        html = f"""
        <div class="section">
            <h2>🏆 Competitive Analysis</h2>
            
            <h3>Market Share Distribution</h3>
        """
        
        for company, share in market_share.items():
            html += f"""
            <div class="competitor">
                <strong>{company.title()}:</strong> {share:.1f}% market share
            </div>
            """
        
        html += f"""
            <h3>Keyword Overlap</h3>
            <div class="stats-grid">
        """
        
        for competitor, overlap in keyword_overlap.items():
            html += f"""
                <div class="stat-card">
                    <div class="stat-value">{overlap:,}</div>
                    <div class="stat-label">Shared with {competitor.title()}</div>
                </div>
            """
        
        html += "</div></div>"
        return html
    
    def _create_keyword_analysis_section(self, keyword_analysis: Dict[str, Any]) -> str:
        """Create keyword analysis section"""
        
        if not keyword_analysis:
            return ""
        
        position_dist = keyword_analysis.get('position_distribution', {})
        best_keywords = keyword_analysis.get('best_keywords', [])
        
        html = f"""
        <div class="section">
            <h2>🎯 Keyword Performance Analysis</h2>
            
            <h3>Position Distribution</h3>
            <div class="stats-grid">
        """
        
        for position_range, count in position_dist.items():
            html += f"""
                <div class="stat-card">
                    <div class="stat-value">{count:,}</div>
                    <div class="stat-label">{position_range.replace('_', ' ').title()}</div>
                </div>
            """
        
        html += "</div>"
        
        # Best performing keywords
        if best_keywords:
            html += """
            <h3>Top Performing Keywords</h3>
            <table>
                <tr><th>Keyword</th><th>Position</th><th>Traffic (%)</th></tr>
            """
            
            for keyword in best_keywords[:10]:
                html += f"""
                <tr>
                    <td>{keyword.get('Keyword', '')}</td>
                    <td>{keyword.get('Position', 0)}</td>
                    <td>{keyword.get('Traffic (%)', 0):.2f}%</td>
                </tr>
                """
            
            html += "</table>"
        
        html += "</div>"
        return html
    
    def _create_opportunity_section(self, opportunity_analysis: Dict[str, Any]) -> str:
        """Create opportunity analysis section"""
        
        if not opportunity_analysis:
            return ""
        
        opportunities = opportunity_analysis.get('high_value_opportunities', [])
        total_gaps = opportunity_analysis.get('total_gap_keywords', 0)
        
        html = f"""
        <div class="section">
            <h2>💡 Opportunity Analysis</h2>
            
            <div class="metric">
                <h3>🎯 Gap Keywords Identified: {total_gaps:,}</h3>
                <p>High-value opportunities: {len(opportunities)}</p>
            </div>
        """
        
        if opportunities:
            html += """
            <h3>Top Opportunities</h3>
            <table>
                <tr><th>Keyword</th><th>Search Volume</th><th>Difficulty</th></tr>
            """
            
            for opp in opportunities[:15]:
                html += f"""
                <tr>
                    <td>{opp.get('Keyword', '')}</td>
                    <td>{opp.get('Volume', 0):,}</td>
                    <td>{opp.get('Keyword Difficulty', 0)}</td>
                </tr>
                """
            
            html += "</table>"
        
        html += "</div>"
        return html
    
    def _create_recommendations_section(self, results: Dict[str, Any]) -> str:
        """Create strategic recommendations section"""
        
        keyword_analysis = results.get('keyword_analysis', {})
        opportunity_analysis = results.get('opportunity_analysis', {})
        competitive = results.get('competitive', {})
        
        html = f"""
        <div class="section">
            <h2>💡 Strategic Recommendations</h2>
            
            <h3>Immediate Actions (0-30 days)</h3>
        """
        
        # Generate recommendations based on analysis
        recommendations = []
        
        # Position-based recommendations
        position_dist = keyword_analysis.get('position_distribution', {})
        if position_dist.get('top_20', 0) > position_dist.get('top_10', 0):
            recommendations.append("Optimize keywords ranking 11-20 to reach page 1")
        
        # Opportunity-based recommendations
        opportunities = opportunity_analysis.get('high_value_opportunities', [])
        if len(opportunities) > 10:
            recommendations.append(f"Target {len(opportunities)} high-value gap keywords")
        
        # Market share recommendations
        market_share = competitive.get('market_share', {})
        lenovo_share = market_share.get('lenovo', 0)
        if lenovo_share < 40:
            recommendations.append("Increase market share through competitive keyword targeting")
        
        for i, rec in enumerate(recommendations, 1):
            html += f'<div class="insight">🚀 {i}. {rec}</div>'
        
        html += """
            <h3>Medium-term Actions (1-3 months)</h3>
            <div class="insight">📈 Develop content strategy for high-volume keywords</div>
            <div class="insight">🎯 Improve SERP feature optimization</div>
            <div class="insight">📊 Monitor competitive movements and respond accordingly</div>
        </div>
        """
        
        return html
    
    def _generate_json_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive JSON report"""
        
        # Add metadata
        enhanced_results = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_period': 'May 19-21, 2025',
                'report_version': '2.0',
                'features': ['core_analysis', 'competitive_comparison', 'opportunities', 'visualizations']
            },
            'analysis_results': results
        }
        
        report_path = self.reports_dir / "enhanced_seo_analysis_data.json"
        
        with open(report_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        return str(report_path)
    
    def _generate_excel_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive Excel report"""
        
        report_path = self.reports_dir / "enhanced_seo_analysis_data.xlsx"
        
        try:
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = results.get('summary', {})
                if summary_data:
                    summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
                    summary_df.to_excel(writer, sheet_name='Summary', index=True)
                
                # Competitive analysis sheet
                competitive_data = results.get('competitive', {})
                if competitive_data:
                    # Market share
                    market_share_df = pd.DataFrame(list(competitive_data.get('market_share', {}).items()), 
                                                 columns=['Company', 'Market_Share'])
                    market_share_df.to_excel(writer, sheet_name='Market_Share', index=False)
                    
                    # Traffic comparison
                    traffic_df = pd.DataFrame(list(competitive_data.get('traffic_comparison', {}).items()), 
                                            columns=['Company', 'Traffic_Share'])
                    traffic_df.to_excel(writer, sheet_name='Traffic_Comparison', index=False)
                
                # Keyword analysis
                keyword_analysis = results.get('keyword_analysis', {})
                if keyword_analysis:
                    # Position distribution
                    position_df = pd.DataFrame(list(keyword_analysis.get('position_distribution', {}).items()), 
                                             columns=['Position_Range', 'Keyword_Count'])
                    position_df.to_excel(writer, sheet_name='Position_Distribution', index=False)
                    
                    # Best keywords
                    best_keywords = keyword_analysis.get('best_keywords', [])
                    if best_keywords:
                        best_df = pd.DataFrame(best_keywords)
                        best_df.to_excel(writer, sheet_name='Best_Keywords', index=False)
                
                # Opportunities
                opportunity_analysis = results.get('opportunity_analysis', {})
                opportunities = opportunity_analysis.get('high_value_opportunities', [])
                if opportunities:
                    opp_df = pd.DataFrame(opportunities)
                    opp_df.to_excel(writer, sheet_name='Opportunities', index=False)
                
                # Metadata sheet
                metadata_df = pd.DataFrame([{
                    'Generated_At': datetime.now(),
                    'Analysis_Period': 'May 19-21, 2025',
                    'Report_Version': '2.0',
                    'Total_Sheets': len(writer.sheets)
                }])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        except Exception as e:
            print(f"⚠️ Error generating Excel report: {e}")
            # Create simple text file as fallback
            with open(report_path.with_suffix('.txt'), 'w') as f:
                f.write(f"Excel Report Generation Failed: {e}\n")
                f.write(f"Data: {json.dumps(results, indent=2, default=str)}")
            return str(report_path.with_suffix('.txt'))
        
        return str(report_path)

    def create_advanced_visualizations(self, results: Dict[str, Any], advanced_metrics: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, str]:
        """Create advanced visualizations based on predictive insights"""
        
        print("📊 Creating advanced visualizations...")
        
        viz_dir = self.reports_dir / "visuals" / "advanced"
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        charts = {}
        
        try:
            # 1. Competitive Threat Radar Chart
            charts['threat_radar'] = self._create_threat_radar_chart(advanced_metrics, viz_dir)
            
            # 2. Traffic Forecast Chart
            if 'traffic_forecast' in predictions:
                charts['traffic_forecast'] = self._create_traffic_forecast_chart(predictions['traffic_forecast'], viz_dir)
            
            # 3. Market Dominance Bubble Chart
            if 'market_dominance' in advanced_metrics:
                charts['market_dominance'] = self._create_market_dominance_chart(advanced_metrics['market_dominance'], viz_dir)
            
            # 4. Opportunity Scoring Matrix
            if 'keyword_opportunities' in predictions:
                charts['opportunity_matrix'] = self._create_opportunity_matrix_chart(predictions['keyword_opportunities'], viz_dir)
            
            # 5. Risk Assessment Dashboard
            if 'risk_assessment' in predictions:
                charts['risk_dashboard'] = self._create_risk_dashboard_chart(predictions['risk_assessment'], viz_dir)
            
            print(f"📊 Generated {len(charts)} advanced visualizations")
            
        except Exception as e:
            print(f"⚠️ Error generating advanced visualizations: {e}")
        
        return charts

    def _create_threat_radar_chart(self, advanced_metrics: Dict[str, Any], viz_dir: Path) -> str:
        """Create competitive threat radar chart"""
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Threat dimensions
            categories = ['Market Share', 'Position Velocity', 'SERP Dominance', 'Traffic Growth', 'Keyword Coverage']
            
            # Get data for each competitor
            competitors = ['Lenovo', 'Dell', 'HP']
            colors = ['#2E86C1', '#E74C3C', '#F39C12']
            
            # Normalize scores to 0-10 scale
            def normalize_score(value, max_val=1.0):
                return min(10, (value / max_val) * 10) if value else 0
            
            # Calculate scores for each competitor
            competitor_scores = {}
            
            market_dominance = advanced_metrics.get('market_dominance', {})
            position_velocity = advanced_metrics.get('position_velocity', {})
            serp_dominance = advanced_metrics.get('serp_dominance', {})
            
            for comp in ['lenovo', 'dell', 'hp']:
                scores = [
                    normalize_score(market_dominance.get(comp, {}).get('market_share', 0), 0.5),
                    normalize_score(abs(position_velocity.get(comp, {}).get('pvi_score', 0)), 3.0),
                    normalize_score(serp_dominance.get(comp, {}).get('dominance_score', 0), 3.0),
                    normalize_score(0.5, 1.0),  # Traffic growth placeholder
                    normalize_score(0.6, 1.0)   # Keyword coverage placeholder
                ]
                competitor_scores[comp] = scores
            
            # Number of variables
            N = len(categories)
            
            # Angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Plot each competitor
            for i, (comp, color) in enumerate(zip(['lenovo', 'dell', 'hp'], colors)):
                values = competitor_scores.get(comp, [0] * N)
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=comp.title(), color=color)
                ax.fill(angles, values, alpha=0.25, color=color)
            
            # Add category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 10)
            
            # Add grid
            ax.grid(True)
            
            # Add legend and title
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            plt.title('Competitive Threat Analysis Radar', size=16, fontweight='bold', pad=20)
            
            chart_path = viz_dir / "competitive_threat_radar.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating threat radar chart: {e}")
            return ""

    def _create_traffic_forecast_chart(self, traffic_forecast: Dict[str, Any], viz_dir: Path) -> str:
        """Create traffic forecast visualization"""
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Historical trend (simulated data points)
            days = np.arange(-7, 31)  # 7 days historical, 30 days forecast
            historical_days = days[days <= 0]
            forecast_days = days[days > 0]
            
            # Simulate historical data based on trend
            slope = traffic_forecast.get('slope', 0)
            base_traffic = 100  # Baseline traffic
            
            historical_traffic = base_traffic + slope * historical_days
            forecast_traffic = base_traffic + slope * forecast_days
            
            # Plot historical data
            plt.plot(historical_days, historical_traffic, 'o-', color='#2E86C1', linewidth=2, label='Historical Data')
            
            # Plot forecast
            plt.plot(forecast_days, forecast_traffic, '--', color='#E74C3C', linewidth=2, label='Forecast')
            
            # Add confidence interval
            confidence = traffic_forecast.get('confidence', 0.5)
            forecast_std = np.std(forecast_traffic) * (1 - confidence)
            
            plt.fill_between(forecast_days, 
                            forecast_traffic - forecast_std, 
                            forecast_traffic + forecast_std, 
                            alpha=0.3, color='#E74C3C', label=f'Confidence Interval ({confidence:.1%})')
            
            # Formatting
            plt.axvline(x=0, color='gray', linestyle=':', alpha=0.7, label='Today')
            plt.xlabel('Days from Today', fontsize=12, fontweight='bold')
            plt.ylabel('Traffic Share (%)', fontsize=12, fontweight='bold')
            plt.title('Traffic Forecast - 30 Day Projection', fontsize=16, fontweight='bold', pad=20)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add forecast summary text
            forecast_change = traffic_forecast.get('forecast_change', 0)
            reliability = traffic_forecast.get('forecast_reliability', 'Medium')
            
            plt.text(0.02, 0.98, f'Forecast Change: {forecast_change:+.1f}%\nReliability: {reliability}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            plt.tight_layout()
            
            chart_path = viz_dir / "traffic_forecast.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating traffic forecast chart: {e}")
            return ""

    def _create_opportunity_matrix_chart(self, opportunities: Dict[str, Any], viz_dir: Path) -> str:
        """Create opportunity scoring matrix"""
        
        try:
            top_opportunities = opportunities.get('top_opportunities', [])
            
            if not top_opportunities:
                return ""
            
            plt.figure(figsize=(14, 10))
            
            # Extract data
            volumes = [opp.get('Volume', 0) for opp in top_opportunities]
            difficulties = [opp.get('Keyword Difficulty', 0) for opp in top_opportunities]
            opportunity_scores = [opp.get('opportunity_score', 0) for opp in top_opportunities]
            success_probs = [opp.get('success_probability', 0) for opp in top_opportunities]
            
            # Create scatter plot with size based on opportunity score
            scatter = plt.scatter(difficulties, volumes, 
                                s=[score * 100 for score in opportunity_scores], 
                                c=success_probs, 
                                cmap='RdYlGn', 
                                alpha=0.7,
                                edgecolors='black',
                                linewidth=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Success Probability', fontsize=12, fontweight='bold')
            
            # Add quadrant lines
            plt.axvline(x=50, color='red', linestyle='--', alpha=0.5)
            plt.axhline(y=np.median(volumes), color='red', linestyle='--', alpha=0.5)
            
            # Add quadrant labels
            max_vol = max(volumes) if volumes else 1000
            plt.text(25, max_vol * 0.9, 'High Volume\nLow Difficulty\n(Sweet Spot)', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            
            plt.text(75, max_vol * 0.9, 'High Volume\nHigh Difficulty\n(Long-term)', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
            
            # Formatting
            plt.xlabel('Keyword Difficulty', fontsize=12, fontweight='bold')
            plt.ylabel('Search Volume', fontsize=12, fontweight='bold')
            plt.title('SEO Opportunity Matrix\n(Size = Opportunity Score, Color = Success Probability)', 
                    fontsize=16, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3)
            
            # Add legend for bubble sizes
            sizes = [min(opportunity_scores), np.median(opportunity_scores), max(opportunity_scores)]
            size_labels = ['Low', 'Medium', 'High']
            
            for size, label in zip(sizes, size_labels):
                plt.scatter([], [], s=size*100, c='gray', alpha=0.6, 
                        edgecolors='black', linewidth=0.5, label=f'{label} Opportunity')
            
            plt.legend(title='Opportunity Score', loc='upper left')
            
            plt.tight_layout()
            
            chart_path = viz_dir / "opportunity_matrix.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating opportunity matrix chart: {e}")
            return ""
