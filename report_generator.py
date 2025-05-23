"""
Enhanced Report Generator with Complete Visualizations
Comprehensive reporting for all analysis components
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

class ReportGenerator:
    """Generate comprehensive reports with all visualizations"""
    
    def __init__(self):
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.reports_dir / "visuals").mkdir(exist_ok=True)
        (self.reports_dir / "visuals" / "advanced").mkdir(exist_ok=True, parents=True)
        (self.reports_dir / "visuals" / "analysis").mkdir(exist_ok=True, parents=True)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_all_reports(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate all comprehensive reports"""
        
        print("📝 Generating comprehensive reports with all analysis...")
        
        reports = {
            'html': self._generate_comprehensive_html_report(analysis_results),
            'json': self._generate_json_report(analysis_results),
            'excel': self._generate_excel_report(analysis_results)
        }
        
        print("✅ All reports generated successfully!")
        return reports
    
    def generate_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive visualization charts"""
        
        print("📊 Generating comprehensive visualizations...")
        
        viz_dir = self.reports_dir / "visuals"
        analysis_dir = viz_dir / "analysis"
        
        charts = {}
        
        try:
            # Core Analysis Charts
            competitive = results.get('competitive', {})
            if 'market_share' in competitive:
                charts['market_share'] = self._create_market_share_chart(competitive['market_share'], viz_dir)
            
            if 'traffic_comparison' in competitive:
                charts['traffic_comparison'] = self._create_traffic_comparison_chart(competitive['traffic_comparison'], viz_dir)
            
            # Keyword Analysis Charts
            keyword_analysis = results.get('keyword_analysis', {})
            if 'position_distribution' in keyword_analysis:
                charts['position_dist'] = self._create_position_distribution_chart(keyword_analysis['position_distribution'], viz_dir)
            
            # Advanced Keyword Analysis Charts
            advanced_keyword = results.get('advanced_keyword_analysis', {})
            if 'intent_distribution' in advanced_keyword:
                charts['intent_distribution'] = self._create_intent_distribution_chart(advanced_keyword['intent_distribution'], analysis_dir)
            
            if 'serp_features' in advanced_keyword:
                charts['serp_features'] = self._create_serp_features_chart(advanced_keyword['serp_features'], analysis_dir)
            
            if 'branded_vs_nonbranded' in advanced_keyword:
                charts['branded_analysis'] = self._create_branded_analysis_chart(advanced_keyword['branded_vs_nonbranded'], analysis_dir)
            
            # Gap Analysis Charts
            gap_analysis = results.get('competitive_gaps', {})
            if 'position_gaps' in gap_analysis:
                charts['position_gaps'] = self._create_position_gaps_chart(gap_analysis['position_gaps'], analysis_dir)
            
            if 'quick_wins' in gap_analysis:
                charts['quick_wins'] = self._create_quick_wins_chart(gap_analysis['quick_wins'], analysis_dir)
            
            # Opportunity Analysis Charts
            opportunity_analysis = results.get('opportunity_analysis', {})
            if opportunity_analysis:
                charts['opportunities'] = self._create_opportunities_chart(opportunity_analysis, viz_dir)
            
            # Advanced Metrics Visualization
            advanced_metrics = results.get('advanced_metrics', {})
            if 'threat_assessment' in advanced_metrics:
                charts['threat_assessment'] = self._create_threat_assessment_chart(advanced_metrics['threat_assessment'], analysis_dir)
            
            print(f"📊 Generated {len(charts)} comprehensive visualizations")
            
        except Exception as e:
            print(f"⚠️ Error generating some visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        return charts
    
    def create_advanced_visualizations(self, results: Dict[str, Any], advanced_metrics: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, str]:
        """Create advanced predictive visualizations"""
        
        print("🔬 Creating advanced predictive visualizations...")
        
        viz_dir = self.reports_dir / "visuals" / "advanced"
        
        charts = {}
        
        try:
            # Advanced visualizations
            if advanced_metrics:
                charts['threat_radar'] = self._create_threat_radar_chart(advanced_metrics, viz_dir)
                
                if 'market_dominance' in advanced_metrics:
                    charts['market_dominance'] = self._create_market_dominance_chart(advanced_metrics['market_dominance'], viz_dir)
                
                if 'position_velocity' in advanced_metrics:
                    charts['velocity_analysis'] = self._create_velocity_analysis_chart(advanced_metrics['position_velocity'], viz_dir)
            
            # Predictive visualizations
            if predictions:
                if 'traffic_forecast' in predictions:
                    charts['traffic_forecast'] = self._create_traffic_forecast_chart(predictions['traffic_forecast'], viz_dir)
                
                if 'keyword_opportunities' in predictions:
                    charts['opportunity_matrix'] = self._create_opportunity_matrix_chart(predictions['keyword_opportunities'], viz_dir)
                
                if 'risk_assessment' in predictions:
                    charts['risk_dashboard'] = self._create_risk_dashboard_chart(predictions['risk_assessment'], viz_dir)
                
                if 'market_share_trajectory' in predictions:
                    charts['market_trajectory'] = self._create_market_trajectory_chart(predictions['market_share_trajectory'], viz_dir)
            
            # Action Plan Visualization
            action_plan = results.get('action_plan', {})
            if action_plan:
                charts['action_priority'] = self._create_action_priority_chart(action_plan, viz_dir)
            
            print(f"🔬 Generated {len(charts)} advanced visualizations")
            
        except Exception as e:
            print(f"⚠️ Error generating advanced visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        return charts
    
    # =========================================================================
    # MISSING VISUALIZATION METHODS
    # =========================================================================
    
    def _create_market_dominance_chart(self, market_dominance: Dict[str, Any], viz_dir: Path) -> str:
        """Create market dominance bubble chart"""
        
        try:
            plt.figure(figsize=(12, 8))
            
            companies = []
            market_shares = []
            dominance_scores = []
            colors = ['#2E86C1', '#E74C3C', '#27AE60']
            
            for i, (company, data) in enumerate(market_dominance.items()):
                companies.append(company.title())
                market_shares.append(data.get('market_share', 0) * 100)  # Convert to percentage
                dominance_scores.append(data.get('dominance_score', 0) * 1000)  # Scale for bubble size
            
            # Create bubble chart
            scatter = plt.scatter(range(len(companies)), market_shares, 
                                s=dominance_scores, alpha=0.7, c=colors[:len(companies)])
            
            # Customize chart
            plt.xticks(range(len(companies)), companies)
            plt.ylabel('Market Share (%)', fontsize=12, fontweight='bold')
            plt.title('Market Dominance Analysis\n(Bubble size = Dominance Score)', fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels
            for i, (company, share, score) in enumerate(zip(companies, market_shares, dominance_scores)):
                plt.annotate(f'{share:.1f}%\nScore: {score/1000:.3f}', 
                           (i, share), 
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', va='bottom', fontweight='bold')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            chart_path = viz_dir / "market_dominance_bubble.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating market dominance chart: {e}")
            return ""
    
    def _create_intent_distribution_chart(self, intent_dist: Dict[str, int], viz_dir: Path) -> str:
        """Create intent distribution donut chart"""
        
        try:
            plt.figure(figsize=(10, 8))
            
            intents = list(intent_dist.keys())
            counts = list(intent_dist.values())
            colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
            
            # Create donut chart
            wedges, texts, autotexts = plt.pie(counts, labels=intents, autopct='%1.1f%%', 
                                              colors=colors, startangle=90, pctdistance=0.85)
            
            # Create center circle for donut effect
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            plt.gca().add_artist(centre_circle)
            
            # Enhance text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.title('Keyword Intent Distribution', fontsize=16, fontweight='bold', pad=20)
            
            # Add center text
            total_keywords = sum(counts)
            plt.text(0, 0, f'{total_keywords:,}\nKeywords', ha='center', va='center', 
                    fontsize=14, fontweight='bold')
            
            plt.axis('equal')
            
            chart_path = viz_dir / "intent_distribution.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating intent distribution chart: {e}")
            return ""
    
    def _create_serp_features_chart(self, serp_data: Dict[str, Any], viz_dir: Path) -> str:
        """Create SERP features analysis chart"""
        
        try:
            feature_breakdown = serp_data.get('feature_breakdown', {})
            if not feature_breakdown:
                return ""
            
            plt.figure(figsize=(12, 8))
            
            # Get top 10 features
            top_features = sorted(feature_breakdown.items(), key=lambda x: x[1], reverse=True)[:10]
            features = [f[0] for f in top_features]
            counts = [f[1] for f in top_features]
            
            # Create horizontal bar chart
            bars = plt.barh(range(len(features)), counts, color='steelblue')
            
            # Customize
            plt.yticks(range(len(features)), features)
            plt.xlabel('Number of Keywords', fontsize=12, fontweight='bold')
            plt.title('SERP Features Presence Analysis', fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{int(width)}', va='center', fontweight='bold')
            
            # Add coverage percentage
            coverage = serp_data.get('coverage_percentage', 0)
            plt.figtext(0.5, 0.02, f'Overall SERP Features Coverage: {coverage:.1f}%', 
                       ha='center', fontsize=10, style='italic')
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            chart_path = viz_dir / "serp_features_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating SERP features chart: {e}")
            return ""
    
    def _create_branded_analysis_chart(self, branded_data: Dict[str, Any], viz_dir: Path) -> str:
        """Create branded vs non-branded analysis chart"""
        
        try:
            plt.figure(figsize=(12, 6))
            
            # Data for comparison
            categories = ['Branded', 'Non-Branded']
            keyword_counts = [branded_data.get('branded_count', 0), branded_data.get('non_branded_count', 0)]
            avg_positions = [branded_data.get('branded_avg_position', 0), branded_data.get('non_branded_avg_position', 0)]
            traffic_shares = [branded_data.get('branded_traffic_share', 0), branded_data.get('non_branded_traffic_share', 0)]
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Keyword counts
            bars1 = ax1.bar(categories, keyword_counts, color=['#3498DB', '#E74C3C'])
            ax1.set_title('Keyword Count Comparison', fontweight='bold')
            ax1.set_ylabel('Number of Keywords')
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
            
            # Average positions (lower is better)
            bars2 = ax2.bar(categories, avg_positions, color=['#2ECC71', '#F39C12'])
            ax2.set_title('Average Position Comparison', fontweight='bold')
            ax2.set_ylabel('Average Position')
            ax2.invert_yaxis()  # Lower positions are better
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='top', fontweight='bold')
            
            # Traffic shares
            bars3 = ax3.bar(categories, traffic_shares, color=['#9B59B6', '#1ABC9C'])
            ax3.set_title('Traffic Share Comparison', fontweight='bold')
            ax3.set_ylabel('Traffic Share (%)')
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Performance efficiency (traffic per keyword)
            efficiency = [traffic_shares[i]/keyword_counts[i] if keyword_counts[i] > 0 else 0 for i in range(2)]
            bars4 = ax4.bar(categories, efficiency, color=['#E67E22', '#34495E'])
            ax4.set_title('Traffic Efficiency (Traffic % per Keyword)', fontweight='bold')
            ax4.set_ylabel('Efficiency Score')
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle('Branded vs Non-Branded Performance Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = viz_dir / "branded_vs_nonbranded_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating branded analysis chart: {e}")
            return ""
    
    def _create_position_gaps_chart(self, position_gaps: List[Dict], viz_dir: Path) -> str:
        """Create position gaps analysis chart"""
        
        try:
            if not position_gaps:
                return ""
            
            plt.figure(figsize=(14, 8))
            
            # Prepare data for top 15 gaps
            top_gaps = position_gaps[:15]
            keywords = [gap['keyword'][:25] + '...' if len(gap['keyword']) > 25 else gap['keyword'] for gap in top_gaps]
            gap_sizes = [gap['gap'] for gap in top_gaps]
            competitors = [gap['competitor'] for gap in top_gaps]
            traffic_potential = [gap.get('traffic_potential', 0) for gap in top_gaps]
            
            # Create color map for competitors
            color_map = {'Dell': '#E74C3C', 'HP': '#3498DB'}
            colors = [color_map.get(comp, '#95A5A6') for comp in competitors]
            
            # Create horizontal bar chart
            bars = plt.barh(range(len(keywords)), gap_sizes, color=colors)
            
            # Customize chart
            plt.yticks(range(len(keywords)), keywords)
            plt.xlabel('Position Gap (Competitor Advantage)', fontsize=12, fontweight='bold')
            plt.title('Top Position Gaps vs Competitors', fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels with traffic potential
            for i, (bar, gap, comp, traffic) in enumerate(zip(bars, gap_sizes, competitors, traffic_potential)):
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{gap} ({comp})\n{traffic:.1f}% traffic', 
                        va='center', fontsize=8, fontweight='bold')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#E74C3C', label='Dell'),
                              Patch(facecolor='#3498DB', label='HP')]
            plt.legend(handles=legend_elements, loc='lower right')
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            chart_path = viz_dir / "position_gaps_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating position gaps chart: {e}")
            return ""
    
    def _create_quick_wins_chart(self, quick_wins: List[Dict], viz_dir: Path) -> str:
        """Create quick wins opportunities chart"""
        
        try:
            if not quick_wins:
                return ""
            
            plt.figure(figsize=(12, 8))
            
            # Separate different types of quick wins
            page_2_wins = [qw for qw in quick_wins if qw.get('type') == 'page_2_keywords']
            gap_wins = [qw for qw in quick_wins if qw.get('type') == 'easy_gap_keywords']
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Page 2 keywords chart
            if page_2_wins:
                keywords = [win['keyword'][:20] + '...' if len(win['keyword']) > 20 else win['keyword'] for win in page_2_wins[:10]]
                positions = [win['current_position'] for win in page_2_wins[:10]]
                traffic = [win.get('traffic_potential', 0) for win in page_2_wins[:10]]
                
                # Create scatter plot
                scatter1 = ax1.scatter(positions, range(len(keywords)), s=[t*10 for t in traffic], 
                                     c=traffic, cmap='YlOrRd', alpha=0.7)
                ax1.set_yticks(range(len(keywords)))
                ax1.set_yticklabels(keywords)
                ax1.set_xlabel('Current Position')
                ax1.set_title('Page 2 Quick Wins\n(Size & Color = Traffic Potential)', fontweight='bold')
                ax1.grid(axis='x', alpha=0.3)
                
                # Add colorbar
                plt.colorbar(scatter1, ax=ax1, label='Traffic Potential (%)')
            
            # Gap keywords chart
            if gap_wins:
                gap_keywords = [win['keyword'][:20] + '...' if len(win['keyword']) > 20 else win['keyword'] for win in gap_wins[:10]]
                volumes = [win.get('volume', 0) for win in gap_wins[:10]]
                difficulties = [win.get('difficulty', 0) for win in gap_wins[:10]]
                
                # Create scatter plot
                scatter2 = ax2.scatter(difficulties, volumes, s=100, c=range(len(gap_keywords)), 
                                     cmap='viridis', alpha=0.7)
                ax2.set_xlabel('Keyword Difficulty')
                ax2.set_ylabel('Search Volume')
                ax2.set_title('Gap Keyword Quick Wins\n(Low Difficulty, High Volume)', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Add keyword labels
                for i, (diff, vol, kw) in enumerate(zip(difficulties, volumes, gap_keywords)):
                    ax2.annotate(kw, (diff, vol), xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.8)
            
            plt.suptitle('Quick Win Opportunities Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = viz_dir / "quick_wins_opportunities.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating quick wins chart: {e}")
            return ""
    
    def _create_threat_assessment_chart(self, threat_data: Dict[str, Any], viz_dir: Path) -> str:
        """Create threat assessment visualization"""
        
        try:
            plt.figure(figsize=(10, 6))
            
            threat_level = threat_data.get('threat_level', 'Unknown')
            active_threats = threat_data.get('active_threats', [])
            
            # Create threat level gauge
            levels = ['Low', 'Medium', 'High']
            colors = ['#2ECC71', '#F39C12', '#E74C3C']
            
            # Current threat level index
            current_index = levels.index(threat_level) if threat_level in levels else 1
            
            # Create bar chart for threat levels
            bars = plt.bar(levels, [1, 1, 1], color=['lightgray' if i != current_index else colors[i] for i in range(3)])
            
            # Highlight current threat level
            bars[current_index].set_color(colors[current_index])
            bars[current_index].set_edgecolor('black')
            bars[current_index].set_linewidth(3)
            
            plt.title(f'Current Threat Level: {threat_level}', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Threat Assessment')
            
            # Add threat details as text
            if active_threats:
                threat_text = "Active Threats:\n" + "\n".join([f"• {threat}" for threat in active_threats[:5]])
                plt.text(0.02, 0.98, threat_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor="lightyellow", alpha=0.8), fontsize=10)
            
            plt.tight_layout()
            
            chart_path = viz_dir / "threat_assessment.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating threat assessment chart: {e}")
            return ""
    
    def _create_velocity_analysis_chart(self, velocity_data: Dict[str, Any], viz_dir: Path) -> str:
        """Create position velocity analysis chart"""
        
        try:
            plt.figure(figsize=(12, 8))
            
            companies = []
            velocities = []
            trends = []
            
            for company, data in velocity_data.items():
                companies.append(company.title())
                velocities.append(data.get('velocity_magnitude', 0))
                trends.append(data.get('trend', 'Stable'))
            
            # Color mapping for trends
            trend_colors = {'Improving': '#2ECC71', 'Declining': '#E74C3C', 'Stable': '#F39C12'}
            colors = [trend_colors.get(trend, '#95A5A6') for trend in trends]
            
            # Create bar chart
            bars = plt.bar(companies, velocities, color=colors)
            
            plt.title('Position Velocity Analysis', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Company')
            plt.ylabel('Velocity Magnitude')
            
            # Add value labels and trend indicators
            for bar, velocity, trend in zip(bars, velocities, trends):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{velocity:.2f}\n({trend})', ha='center', va='bottom', fontweight='bold')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#2ECC71', label='Improving'),
                              Patch(facecolor='#E74C3C', label='Declining'),
                              Patch(facecolor='#F39C12', label='Stable')]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            chart_path = viz_dir / "velocity_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating velocity analysis chart: {e}")
            return ""
    
    def _create_market_trajectory_chart(self, trajectory_data: Dict[str, Any], viz_dir: Path) -> str:
        """Create market share trajectory chart"""
        
        try:
            plt.figure(figsize=(10, 6))
            
            current_share = trajectory_data.get('current_share', 0)
            projected_change = trajectory_data.get('projected_3_month_change', 0)
            projected_share = trajectory_data.get('projected_share', current_share)
            
            # Create trajectory line
            months = ['Current', '1 Month', '2 Months', '3 Months']
            shares = [current_share, 
                     current_share + projected_change/3, 
                     current_share + 2*projected_change/3, 
                     projected_share]
            
            plt.plot(months, shares, 'o-', linewidth=3, markersize=8, color='#3498DB')
            plt.fill_between(months, shares, alpha=0.3, color='#3498DB')
            
            plt.title('Market Share Trajectory Forecast', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Market Share (%)')
            plt.xlabel('Time Period')
            
            # Add value labels
            for month, share in zip(months, shares):
                plt.annotate(f'{share:.1f}%', (month, share), xytext=(0, 10), 
                           textcoords='offset points', ha='center', va='bottom', fontweight='bold')
            
            # Add trend indicator
            trend_color = '#2ECC71' if projected_change > 0 else '#E74C3C' if projected_change < 0 else '#F39C12'
            trend_text = f'Projected Change: {projected_change:+.1f}%'
            plt.text(0.02, 0.98, trend_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor=trend_color, alpha=0.7), fontsize=12, fontweight='bold')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            chart_path = viz_dir / "market_trajectory.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating market trajectory chart: {e}")
            return ""
    
    def _create_action_priority_chart(self, action_plan: Dict[str, Any], viz_dir: Path) -> str:
        """Create action priority matrix chart"""
        
        try:
            prioritized_actions = action_plan.get('prioritized_actions', [])
            if not prioritized_actions:
                return ""
            
            plt.figure(figsize=(14, 10))
            
            # Prepare data
            top_actions = prioritized_actions[:15]  # Top 15 actions
            actions = [action['action'][:30] + '...' if len(action['action']) > 30 else action['action'] for action in top_actions]
            priorities = [action.get('priority_score', 0) for action in top_actions]
            categories = [action.get('category', 'other').replace('_', ' ').title() for action in top_actions]
            
            # Color mapping for categories
            category_colors = {
                'Immediate Actions': '#E74C3C',
                'Defensive Actions': '#F39C12', 
                'Offensive Actions': '#2ECC71',
                'Short Term Actions': '#3498DB',
                'Long Term Actions': '#9B59B6'
            }
            colors = [category_colors.get(cat, '#95A5A6') for cat in categories]
            
            # Create horizontal bar chart
            bars = plt.barh(range(len(actions)), priorities, color=colors)
            
            plt.yticks(range(len(actions)), actions)
            plt.xlabel('Priority Score', fontsize=12, fontweight='bold')
            plt.title('Strategic Action Priority Matrix', fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels
            for bar, priority, category in zip(bars, priorities, categories):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{priority:.1f}\n({category})', va='center', fontsize=8)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color, label=cat) 
                              for cat, color in category_colors.items()]
            plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            chart_path = viz_dir / "action_priority_matrix.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating action priority chart: {e}")
            return ""
    
    # =========================================================================
    # EXISTING METHODS (Enhanced)
    # =========================================================================
    
    def _create_market_share_chart(self, market_share: Dict[str, float], viz_dir: Path) -> str:
        """Create enhanced market share pie chart"""
        
        plt.figure(figsize=(10, 8))
        
        companies = list(market_share.keys())
        shares = list(market_share.values())
        colors = ['#2E86C1', '#F39C12', '#27AE60']
        
        # Create pie chart with enhanced styling
        wedges, texts, autotexts = plt.pie(shares, labels=companies, autopct='%1.1f%%', 
                                          colors=colors, startangle=90, explode=(0.05, 0, 0),
                                          shadow=True, textprops={'fontsize': 12})
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        for text in texts:
            text.set_fontsize(14)
            text.set_fontweight('bold')
        
        plt.title('Market Share Distribution by Organic Traffic', fontsize=16, fontweight='bold', pad=20)
        
        # Add total traffic info
        total_traffic = sum(shares)
        plt.figtext(0.5, 0.02, f'Total Traffic Analyzed: {total_traffic:.1f}%', 
                   ha='center', fontsize=10, style='italic')
        
        chart_path = viz_dir / "market_share_distribution.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(chart_path)
    
    def _create_position_distribution_chart(self, position_dist: Dict[str, int], viz_dir: Path) -> str:
        """Create enhanced position distribution chart"""
        
        plt.figure(figsize=(12, 8))
        
        positions = list(position_dist.keys())
        counts = list(position_dist.values())
        
        # Enhanced color scheme
        colors = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#95A5A6']
        
        bars = plt.bar(positions, counts, color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Lenovo Keyword Position Distribution Analysis', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Position Range', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Keywords', fontsize=12, fontweight='bold')
        
        # Enhanced grid
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add percentage labels
        total_keywords = sum(counts)
        if total_keywords > 0:
            for i, (pos, count) in enumerate(zip(positions, counts)):
                percentage = (count / total_keywords) * 100
                plt.text(i, count + max(counts)*0.05, f'({percentage:.1f}%)', 
                        ha='center', va='bottom', fontsize=10, style='italic', fontweight='bold')
        
        plt.tight_layout()
        
        chart_path = viz_dir / "position_distribution_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(chart_path)
    
    def _create_traffic_comparison_chart(self, traffic_comparison: Dict[str, float], viz_dir: Path) -> str:
        """Create enhanced traffic comparison chart"""
        
        plt.figure(figsize=(12, 8))
        
        companies = list(traffic_comparison.keys())
        traffic = list(traffic_comparison.values())
        colors = ['#3498DB', '#E74C3C', '#2ECC71']
        
        bars = plt.bar(companies, traffic, color=colors, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(traffic)*0.01,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.title('Traffic Share Comparison Analysis', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Company', fontsize=12, fontweight='bold')
        plt.ylabel('Traffic Share (%)', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add total and percentages
        total_traffic = sum(traffic)
        for i, (company, t) in enumerate(zip(companies, traffic)):
            percentage = (t / total_traffic) * 100 if total_traffic > 0 else 0
            plt.text(i, t/2, f'{percentage:.1f}%\nof total', ha='center', va='center', 
                    fontweight='bold', color='white', fontsize=10)
        
        plt.tight_layout()
        
        chart_path = viz_dir / "traffic_comparison_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(chart_path)
    
    def _create_opportunities_chart(self, opportunity_analysis: Dict[str, Any], viz_dir: Path) -> str:
        """Create enhanced opportunities analysis chart"""
        
        opportunities = opportunity_analysis.get('high_value_opportunities', [])
        if not opportunities:
            return ""
        
        plt.figure(figsize=(14, 10))
        
        # Extract data
        volumes = [opp.get('Volume', 0) for opp in opportunities]
        difficulties = [opp.get('Keyword Difficulty', 0) for opp in opportunities]
        keywords = [opp.get('Keyword', '')[:15] + '...' if len(opp.get('Keyword', '')) > 15 else opp.get('Keyword', '') for opp in opportunities]
        
        # Create enhanced scatter plot
        scatter = plt.scatter(difficulties, volumes, s=150, alpha=0.7, 
                            c=range(len(opportunities)), cmap='viridis',
                            edgecolors='black', linewidth=1)
        
        # Add labels for top opportunities
        for i, (diff, vol, keyword) in enumerate(zip(difficulties[:10], volumes[:10], keywords[:10])):
            plt.annotate(keyword, (diff, vol), xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        plt.title('SEO Opportunities Analysis: Volume vs Difficulty', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Keyword Difficulty', fontsize=12, fontweight='bold')
        plt.ylabel('Search Volume', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add ideal quadrant annotation
        plt.axvline(x=30, color='red', linestyle='--', alpha=0.5, linewidth=2)
        plt.text(15, max(volumes)*0.9, 'Low Difficulty\n(Easier to Rank)', ha='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
                fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Opportunity Ranking', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        chart_path = viz_dir / "opportunities_analysis_enhanced.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(chart_path)
    
    # =========================================================================
    # COMPREHENSIVE HTML REPORT GENERATION
    # =========================================================================
    
    def _generate_comprehensive_html_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report with ALL analysis"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive SEO Competitive Intelligence Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f8f9fa; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ background: white; margin: 20px 0; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ background: #e8f5e8; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #28a745; }}
                .competitor {{ background: #fff3cd; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #ffc107; }}
                .insight {{ background: #d1ecf1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #17a2b8; }}
                .warning {{ background: #f8d7da; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #dc3545; }}
                .success {{ background: #d4edda; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #28a745; }}
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
                .two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .action-item {{ background: #e7f3ff; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #007bff; }}
                .priority-high {{ border-left-color: #dc3545; background: #f8d7da; }}
                .priority-medium {{ border-left-color: #ffc107; background: #fff3cd; }}
                .priority-low {{ border-left-color: #28a745; background: #d4edda; }}
                .nav-menu {{ background: #343a40; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .nav-menu a {{ color: white; text-decoration: none; margin: 0 15px; }}
                .nav-menu a:hover {{ color: #ffc107; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Comprehensive SEO Competitive Intelligence Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Analysis Period:</strong> May 19-21, 2025</p>
                <p><strong>Features:</strong> Complete Analysis • Advanced Metrics • Predictive Analytics • Strategic Planning</p>
            </div>
            
            <div class="nav-menu">
                <a href="#executive-summary">Executive Summary</a>
                <a href="#data-quality">Data Quality</a>
                <a href="#core-analysis">Core Analysis</a>
                <a href="#advanced-analysis">Advanced Analysis</a>
                <a href="#competitive-analysis">Competitive Analysis</a>
                <a href="#gap-analysis">Gap Analysis</a>
                <a href="#predictive-analytics">Predictive Analytics</a>
                <a href="#strategic-actions">Strategic Actions</a>
            </div>
            
            {self._create_comprehensive_executive_summary(results)}
            {self._create_data_quality_section_html(results.get('data_validation', {}))}
            {self._create_core_analysis_section_html(results)}
            {self._create_advanced_analysis_section_html(results.get('advanced_keyword_analysis', {}))}
            {self._create_comprehensive_competitive_section(results.get('competitive', {}), results.get('advanced_metrics', {}))}
            {self._create_gap_analysis_section_html(results.get('competitive_gaps', {}))}
            {self._create_predictive_analytics_section_html(results.get('predictions', {}), results.get('advanced_metrics', {}))}
            {self._create_strategic_actions_section_html(results.get('action_plan', {}))}
            {self._create_recommendations_section_html(results)}
        </body>
        </html>
        """
        
        report_path = self.reports_dir / "comprehensive_seo_intelligence_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(report_path)
    
    def _create_comprehensive_executive_summary(self, results: Dict[str, Any]) -> str:
        """Create comprehensive executive summary"""
        
        summary = results.get('summary', {})
        competitive = results.get('competitive', {})
        advanced_metrics = results.get('advanced_metrics', {})
        predictions = results.get('predictions', {})
        
        lenovo_data = summary.get('lenovo', {})
        market_share = competitive.get('market_share', {})
        
        html = f"""
        <div class="section" id="executive-summary">
            <h2>📈 Executive Summary</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{lenovo_data.get('total_keywords', 0):,}</div>
                    <div class="stat-label">Total Keywords Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{lenovo_data.get('avg_position', 0):.1f}</div>
                    <div class="stat-label">Average Position</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{lenovo_data.get('total_traffic', 0):.1f}%</div>
                    <div class="stat-label">Total Traffic Share</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{market_share.get('lenovo', 0):.1f}%</div>
                    <div class="stat-label">Market Share</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{lenovo_data.get('top_10_count', 0):,}</div>
                    <div class="stat-label">Top 10 Rankings</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{advanced_metrics.get('threat_assessment', {}).get('threat_level', 'Unknown')}</div>
                    <div class="stat-label">Threat Level</div>
                </div>
            </div>
            
            <div class="two-column">
                <div>
                    <h3>Key Performance Indicators</h3>
                    <div class="metric">
                        <h4>🎯 Position Performance</h4>
                        <p><strong>Top 3 Rankings:</strong> {lenovo_data.get('top_3_count', 0):,} keywords</p>
                        <p><strong>Page 1 Rankings:</strong> {lenovo_data.get('page_1_count', 0):,} keywords</p>
                        <p><strong>Page 2 Rankings:</strong> {lenovo_data.get('page_2_count', 0):,} keywords</p>
                    </div>
                </div>
                <div>
                    <h3>Strategic Status</h3>
                    <div class="metric">
                        <h4>🏆 Competitive Position</h4>
                        <p><strong>Status:</strong> {advanced_metrics.get('market_dominance', {}).get('lenovo', {}).get('competitive_status', 'Unknown')}</p>
                        <p><strong>Market Share:</strong> {market_share.get('lenovo', 0):.1f}%</p>
                        <p><strong>Traffic Risk:</strong> {advanced_metrics.get('traffic_concentration_risk', 'Unknown')}</p>
                    </div>
                </div>
            </div>
        """
        
        # Add predictive insights if available
        if predictions:
            forecast_change = predictions.get('traffic_forecast', {}).get('forecast_change', 0)
            risk_level = predictions.get('risk_assessment', {}).get('overall_risk_level', 'Unknown')
            
            html += f"""
            <div class="two-column">
                <div>
                    <h3>Predictive Insights</h3>
                    <div class="{'success' if forecast_change > 0 else 'warning' if forecast_change < 0 else 'insight'}">
                        <h4>📈 30-Day Forecast</h4>
                        <p><strong>Traffic Change:</strong> {forecast_change:+.1f}%</p>
                        <p><strong>Reliability:</strong> {predictions.get('traffic_forecast', {}).get('forecast_reliability', 'Unknown')}</p>
                    </div>
                </div>
                <div>
                    <h3>Risk Assessment</h3>
                    <div class="{'warning' if risk_level == 'High' else 'insight'}">
                        <h4>🛡️ Overall Risk Level: {risk_level}</h4>
                        <p><strong>Risk Factors:</strong> {len(predictions.get('risk_assessment', {}).get('identified_risks', []))}</p>
                        <p><strong>Active Threats:</strong> {len(predictions.get('competitive_threats', []))}</p>
                    </div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _create_data_quality_section_html(self, validation_data: Dict[str, Any]) -> str:
        """Create data quality section for HTML report"""
        
        if not validation_data:
            return ""
        
        summary = validation_data.get('summary', {})
        quality_score = summary.get('average_quality_score', 0)
        
        quality_color = 'success' if quality_score > 0.8 else 'warning' if quality_score > 0.6 else 'danger'
        
        html = f"""
        <div class="section" id="data-quality">
            <h2>📊 Data Quality Assessment</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" style="color: {'#28a745' if quality_score > 0.8 else '#ffc107' if quality_score > 0.6 else '#dc3545'}">{quality_score:.2f}</div>
                    <div class="stat-label">Overall Quality Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('total_datasets', 0)}</div>
                    <div class="stat-label">Datasets Validated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('total_rows', 0):,}</div>
                    <div class="stat-label">Total Records</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('overall_status', 'Unknown').title()}</div>
                    <div class="stat-label">Quality Status</div>
                </div>
            </div>
        """
        
        # Add dataset-specific quality metrics
        dataset_results = {k: v for k, v in validation_data.items() if k != 'summary'}
        if dataset_results:
            html += """
            <h3>Dataset Quality Breakdown</h3>
            <table>
                <tr><th>Dataset</th><th>Status</th><th>Rows</th><th>Quality Score</th><th>Issues</th></tr>
            """
            
            for dataset, data in dataset_results.items():
                status_color = 'highlight' if data.get('status') in ['excellent', 'good'] else ''
                html += f"""
                <tr class="{status_color}">
                    <td><strong>{dataset.title()}</strong></td>
                    <td>{data.get('status', 'Unknown').title()}</td>
                    <td>{data.get('row_count', 0):,}</td>
                    <td>{data.get('quality_score', 0):.2f}</td>
                    <td>{len(data.get('issues', []))}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            html += "<h3>Quality Improvement Recommendations</h3>"
            for rec in recommendations:
                html += f'<div class="insight">💡 {rec}</div>'
        
        html += "</div>"
        return html
    
    def _create_core_analysis_section_html(self, results: Dict[str, Any]) -> str:
        """Create core analysis section"""
        
        summary = results.get('summary', {})
        keyword_analysis = results.get('keyword_analysis', {})
        opportunity_analysis = results.get('opportunity_analysis', {})
        
        html = f"""
        <div class="section" id="core-analysis">
            <h2>🔍 Core SEO Analysis</h2>
            
            <div class="two-column">
                <div>
                    <h3>Performance Overview</h3>
        """
        
        # Add performance metrics for each company
        for company, data in summary.items():
            if company == 'gap_keywords':
                continue
            
            html += f"""
            <div class="metric">
                <h4>📊 {company.title()} Performance</h4>
                <p><strong>Keywords:</strong> {data.get('total_keywords', 0):,}</p>
                <p><strong>Avg Position:</strong> {data.get('avg_position', 0):.1f}</p>
                <p><strong>Traffic:</strong> {data.get('total_traffic', 0):.1f}%</p>
                <p><strong>Top 10:</strong> {data.get('top_10_count', 0):,}</p>
            </div>
            """
        
        html += """
                </div>
                <div>
                    <h3>Position Analysis</h3>
        """
        
        # Add position distribution if available
        position_dist = keyword_analysis.get('position_distribution', {})
        if position_dist:
            html += """
            <div class="metric">
                <h4>🎯 Position Distribution</h4>
            """
            for position_range, count in position_dist.items():
                html += f"<p><strong>{position_range.replace('_', ' ').title()}:</strong> {count:,} keywords</p>"
            
            html += "</div>"
        
        html += """
                </div>
            </div>
        """
        
        # Add top performing keywords
        best_keywords = keyword_analysis.get('best_keywords', [])
        if best_keywords:
            html += """
            <h3>🏆 Top Performing Keywords</h3>
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
        
        # Add opportunity overview
        if opportunity_analysis:
            total_gaps = opportunity_analysis.get('total_gap_keywords', 0)
            opportunities = opportunity_analysis.get('high_value_opportunities', [])
            
            html += f"""
            <h3>💡 Opportunity Overview</h3>
            <div class="metric">
                <h4>🎯 Gap Analysis Summary</h4>
                <p><strong>Total Gap Keywords:</strong> {total_gaps:,}</p>
                <p><strong>High-Value Opportunities:</strong> {len(opportunities)}</p>
                <p><strong>Avg Difficulty:</strong> {opportunity_analysis.get('avg_difficulty', 0):.1f}</p>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _create_advanced_analysis_section_html(self, advanced_data: Dict[str, Any]) -> str:
        """Create advanced analysis section"""
        
        if not advanced_data:
            return ""
        
        html = f"""
        <div class="section" id="advanced-analysis">
            <h2>🎯 Advanced Keyword Analysis</h2>
        """
        
        # Intent distribution
        intent_dist = advanced_data.get('intent_distribution', {})
        if intent_dist:
            html += """
            <h3>Search Intent Distribution</h3>
            <div class="stats-grid">
            """
            for intent, count in intent_dist.items():
                html += f"""
                <div class="stat-card">
                    <div class="stat-value">{count:,}</div>
                    <div class="stat-label">{intent.title()} Keywords</div>
                </div>
                """
            html += "</div>"
        
        # SERP features analysis
        serp_features = advanced_data.get('serp_features', {})
        if serp_features:
            coverage = serp_features.get('coverage_percentage', 0)
            total_features = serp_features.get('total_with_features', 0)
            
            html += f"""
            <div class="two-column">
                <div>
                    <h3>SERP Features Analysis</h3>
                    <div class="metric">
                        <h4>🔍 Coverage Overview</h4>
                        <p><strong>Coverage:</strong> {coverage:.1f}%</p>
                        <p><strong>Keywords with Features:</strong> {total_features:,}</p>
                    </div>
                </div>
                <div>
                    <h3>Top Features</h3>
                    <div class="metric">
                        <h4>📊 Feature Breakdown</h4>
            """
            
            top_features = serp_features.get('top_features', [])
            for feature, count in top_features[:5]:
                html += f"<p><strong>{feature}:</strong> {count} keywords</p>"
            
            html += """
                    </div>
                </div>
            </div>
            """
        
        # Branded vs non-branded analysis
        branded_analysis = advanced_data.get('branded_vs_nonbranded', {})
        if branded_analysis:
            html += f"""
            <h3>Branded vs Non-Branded Performance</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{branded_analysis.get('branded_count', 0):,}</div>
                    <div class="stat-label">Branded Keywords</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{branded_analysis.get('non_branded_count', 0):,}</div>
                    <div class="stat-label">Non-Branded Keywords</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{branded_analysis.get('branded_avg_position', 0):.1f}</div>
                    <div class="stat-label">Branded Avg Position</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{branded_analysis.get('non_branded_avg_position', 0):.1f}</div>
                    <div class="stat-label">Non-Branded Avg Position</div>
                </div>
            </div>
            """
            
            # Add insights
            insights = branded_analysis.get('insights', [])
            if insights:
                for insight in insights:
                    html += f'<div class="insight">💡 {insight}</div>'
        
        html += "</div>"
        return html
    
    def _create_comprehensive_competitive_section(self, competitive_data: Dict[str, Any], advanced_metrics: Dict[str, Any]) -> str:
        """Create comprehensive competitive analysis section"""
        
        html = f"""
        <div class="section" id="competitive-analysis">
            <h2>🏆 Comprehensive Competitive Analysis</h2>
        """
        
        # Market share and traffic comparison
        market_share = competitive_data.get('market_share', {})
        traffic_comparison = competitive_data.get('traffic_comparison', {})
        
        if market_share:
            html += """
            <h3>Market Share Analysis</h3>
            <div class="stats-grid">
            """
            for company, share in market_share.items():
                html += f"""
                <div class="stat-card">
                    <div class="stat-value">{share:.1f}%</div>
                    <div class="stat-label">{company.title()} Market Share</div>
                </div>
                """
            html += "</div>"
        
        # Advanced competitive metrics
        if advanced_metrics:
            threat_assessment = advanced_metrics.get('threat_assessment', {})
            market_dominance = advanced_metrics.get('market_dominance', {})
            
            if threat_assessment:
                threat_level = threat_assessment.get('threat_level', 'Unknown')
                active_threats = threat_assessment.get('active_threats', [])
                
                html += f"""
                <h3>Threat Assessment</h3>
                <div class="{'warning' if threat_level == 'High' else 'insight'}">
                    <h4>🚨 Current Threat Level: {threat_level}</h4>
                    <p><strong>Active Threats:</strong> {len(active_threats)}</p>
                """
                
                if active_threats:
                    html += "<p><strong>Threat Details:</strong></p><ul>"
                    for threat in active_threats[:3]:
                        html += f"<li>{threat}</li>"
                    html += "</ul>"
                
                html += "</div>"
            
            if market_dominance:
                html += """
                <h3>Market Dominance Analysis</h3>
                <table>
                    <tr><th>Company</th><th>Market Share</th><th>Dominance Score</th><th>Status</th></tr>
                """
                
                for company, data in market_dominance.items():
                    status = data.get('competitive_status', 'Unknown')
                    share = data.get('market_share', 0) * 100
                    score = data.get('dominance_score', 0)
                    
                    row_class = 'highlight' if company == 'lenovo' else ''
                    html += f"""
                    <tr class="{row_class}">
                        <td><strong>{company.title()}</strong></td>
                        <td>{share:.1f}%</td>
                        <td>{score:.3f}</td>
                        <td>{status}</td>
                    </tr>
                    """
                
                html += "</table>"
        
        # Keyword overlap analysis
        keyword_overlap = competitive_data.get('keyword_overlap', {})
        if keyword_overlap:
            html += """
            <h3>Keyword Overlap Analysis</h3>
            <div class="two-column">
            """
            
            for competitor, overlap in keyword_overlap.items():
                html += f"""
                <div class="competitor">
                    <h4>📊 Shared with {competitor.title()}</h4>
                    <p><strong>Common Keywords:</strong> {overlap:,}</p>
                </div>
                """
            
            html += "</div>"
        
        html += "</div>"
        return html
    
    def _create_gap_analysis_section_html(self, gap_data: Dict[str, Any]) -> str:
        """Create gap analysis section"""
        
        if not gap_data:
            return ""
        
        html = f"""
        <div class="section" id="gap-analysis">
            <h2>📉 Competitive Gap Analysis</h2>
        """
        
        # Position gaps
        position_gaps = gap_data.get('position_gaps', [])
        quick_wins = gap_data.get('quick_wins', [])
        
        if position_gaps:
            html += f"""
            <h3>Position Gaps ({len(position_gaps)} identified)</h3>
            <table>
                <tr><th>Keyword</th><th>Lenovo Position</th><th>Competitor</th><th>Competitor Position</th><th>Gap</th><th>Priority</th></tr>
            """
            
            for gap in position_gaps[:10]:
                priority_class = 'priority-high' if gap.get('priority') == 'high' else 'priority-medium'
                html += f"""
                <tr>
                    <td>{gap['keyword']}</td>
                    <td>{gap['lenovo_position']}</td>
                    <td>{gap['competitor']}</td>
                    <td>{gap['competitor_position']}</td>
                    <td class="{priority_class}">{gap['gap']}</td>
                    <td>{gap.get('priority', 'medium').title()}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Quick wins
        if quick_wins:
            html += f"""
            <h3>Quick Win Opportunities ({len(quick_wins)} identified)</h3>
            <div class="two-column">
            """
            
            page_2_wins = [qw for qw in quick_wins if qw.get('type') == 'page_2_keywords']
            gap_wins = [qw for qw in quick_wins if qw.get('type') == 'easy_gap_keywords']
            
            if page_2_wins:
                html += """
                <div>
                    <h4>⚡ Page 2 Keywords</h4>
                """
                for win in page_2_wins[:5]:
                    html += f"""
                    <div class="success">
                        <strong>{win['keyword']}</strong><br>
                        Position: {win['current_position']} | Traffic: {win.get('traffic_potential', 0):.1f}%
                    </div>
                    """
                html += "</div>"
            
            if gap_wins:
                html += """
                <div>
                    <h4>🎯 Easy Gap Keywords</h4>
                """
                for win in gap_wins[:5]:
                    html += f"""
                    <div class="success">
                        <strong>{win['keyword']}</strong><br>
                        Volume: {win.get('volume', 0):,} | Difficulty: {win.get('difficulty', 0)}
                    </div>
                    """
                html += "</div>"
            
            html += "</div>"
        
        html += "</div>"
        return html
    
    def _create_predictive_analytics_section_html(self, predictions: Dict[str, Any], advanced_metrics: Dict[str, Any]) -> str:
        """Create predictive analytics section"""
        
        if not predictions:
            return ""
        
        html = f"""
        <div class="section" id="predictive-analytics">
            <h2>🔮 Predictive Analytics & Forecasting</h2>
        """
        
        # Traffic forecast
        traffic_forecast = predictions.get('traffic_forecast', {})
        if traffic_forecast:
            forecast_change = traffic_forecast.get('forecast_change', 0)
            reliability = traffic_forecast.get('forecast_reliability', 'Unknown')
            
            forecast_class = 'success' if forecast_change > 0 else 'warning' if forecast_change < 0 else 'insight'
            
            html += f"""
            <h3>Traffic Forecast (30-Day Projection)</h3>
            <div class="{forecast_class}">
                <h4>📈 Predicted Change: {forecast_change:+.1f}%</h4>
                <p><strong>Forecast Reliability:</strong> {reliability}</p>
                <p><strong>Current Trend:</strong> {traffic_forecast.get('current_trend', 'Unknown')}</p>
            </div>
            """
        
        # Market share trajectory
        market_trajectory = predictions.get('market_share_trajectory', {})
        if market_trajectory:
            current_share = market_trajectory.get('current_share', 0)
            projected_change = market_trajectory.get('projected_3_month_change', 0)
            projected_share = market_trajectory.get('projected_share', 0)
            
            html += f"""
            <h3>Market Share Trajectory</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{current_share:.1f}%</div>
                    <div class="stat-label">Current Share</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{projected_change:+.1f}%</div>
                    <div class="stat-label">3-Month Change</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{projected_share:.1f}%</div>
                    <div class="stat-label">Projected Share</div>
                </div>
            </div>
            """
        
        # Competitive threats
        competitive_threats = predictions.get('competitive_threats', [])
        if competitive_threats:
            html += f"""
            <h3>Competitive Threat Predictions ({len(competitive_threats)} active)</h3>
            """
            
            for threat in competitive_threats:
                threat_class = 'warning' if threat.get('threat_level') == 'Critical' else 'insight'
                html += f"""
                <div class="{threat_class}">
                    <h4>⚠️ {threat['competitor'].title()} - {threat['threat_level']} Threat</h4>
                    <p><strong>Impact:</strong> {threat['predicted_impact']}</p>
                    <p><strong>Timeline:</strong> {threat['timeline']}</p>
                    <p><strong>Recommendation:</strong> {threat['recommendation']}</p>
                </div>
                """
        
        # Risk assessment
        risk_assessment = predictions.get('risk_assessment', {})
        if risk_assessment:
            risk_level = risk_assessment.get('overall_risk_level', 'Unknown')
            identified_risks = risk_assessment.get('identified_risks', [])
            
            risk_class = 'warning' if risk_level == 'High' else 'insight'
            
            html += f"""
            <h3>Risk Assessment</h3>
            <div class="{risk_class}">
                <h4>🛡️ Overall Risk Level: {risk_level}</h4>
                <p><strong>Risk Factors Identified:</strong> {len(identified_risks)}</p>
            """
            
            if identified_risks:
                html += "<h5>Key Risk Factors:</h5><ul>"
                for risk in identified_risks[:3]:
                    html += f"<li><strong>{risk['risk_type']}:</strong> {risk['description']}</li>"
                html += "</ul>"
            
            html += "</div>"
        
        html += "</div>"
        return html
    
    def _create_strategic_actions_section_html(self, action_plan: Dict[str, Any]) -> str:
        """Create strategic actions section"""
        
        if not action_plan:
            return ""
        
        html = f"""
        <div class="section" id="strategic-actions">
            <h2>📋 Strategic Action Plan</h2>
        """
        
        # Action categories
        immediate_actions = action_plan.get('immediate_actions', [])
        defensive_actions = action_plan.get('defensive_actions', [])
        offensive_actions = action_plan.get('offensive_actions', [])
        
        # Overview
        html += f"""
        <h3>Action Plan Overview</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(immediate_actions)}</div>
                <div class="stat-label">Immediate Actions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(defensive_actions)}</div>
                <div class="stat-label">Defensive Actions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(offensive_actions)}</div>
                <div class="stat-label">Offensive Actions</div>
            </div>
        </div>
        """
        
        # Prioritized actions
        prioritized_actions = action_plan.get('prioritized_actions', [])
        if prioritized_actions:
            html += """
            <h3>Top Priority Actions</h3>
            """
            
            for i, action in enumerate(prioritized_actions[:10], 1):
                priority_score = action.get('priority_score', 0)
                category = action.get('category', 'unknown').replace('_', ' ').title()
                
                if priority_score > 8:
                    priority_class = 'priority-high'
                elif priority_score > 6:
                    priority_class = 'priority-medium'
                else:
                    priority_class = 'priority-low'
                
                html += f"""
                <div class="action-item {priority_class}">
                    <h4>{i}. {action['action']}</h4>
                    <p><strong>Category:</strong> {category}</p>
                    <p><strong>Expected Impact:</strong> {action.get('expected_impact', 'TBD')}</p>
                    <p><strong>Priority Score:</strong> {priority_score:.1f}</p>
                    <p><strong>Resources:</strong> {', '.join(action.get('resources_needed', []))}</p>
                </div>
                """
        
        # Resource requirements
        resource_reqs = action_plan.get('resource_requirements', {})
        if resource_reqs:
            most_needed = resource_reqs.get('most_needed_resources', [])
            
            html += """
            <h3>Resource Requirements</h3>
            <div class="metric">
                <h4>🔧 Resource Allocation</h4>
            """
            
            html += f"<p><strong>Total Actions:</strong> {resource_reqs.get('total_actions', 0)}</p>"
            html += f"<p><strong>Immediate Priority:</strong> {resource_reqs.get('immediate_priority_actions', 0)}</p>"
            
            if most_needed:
                html += "<p><strong>Most Needed Resources:</strong></p><ul>"
                for resource, count in most_needed[:5]:
                    html += f"<li>{resource}: {count} actions</li>"
                html += "</ul>"
            
            html += "</div>"
        
        html += "</div>"
        return html
    
    def _create_recommendations_section_html(self, results: Dict[str, Any]) -> str:
        """Create comprehensive recommendations section"""
        
        html = f"""
        <div class="section">
            <h2>💡 Strategic Recommendations & Next Steps</h2>
            
            <h3>Immediate Priorities (0-30 days)</h3>
        """
        
        # Generate comprehensive recommendations based on all analysis
        immediate_recommendations = []
        medium_term_recommendations = []
        long_term_recommendations = []
        
        # From gap analysis
        gap_data = results.get('competitive_gaps', {})
        quick_wins = gap_data.get('quick_wins', [])
        if quick_wins:
            page_2_count = len([qw for qw in quick_wins if qw.get('type') == 'page_2_keywords'])
            if page_2_count > 0:
                immediate_recommendations.append(f"Optimize {page_2_count} page 2 keywords for quick page 1 wins")
        
        # From advanced analysis
        advanced_data = results.get('advanced_keyword_analysis', {})
        serp_features = advanced_data.get('serp_features', {})
        if serp_features.get('coverage_percentage', 0) < 50:
            immediate_recommendations.append("Improve SERP features optimization - currently low coverage")
        
        # From predictions
        predictions = results.get('predictions', {})
        risk_assessment = predictions.get('risk_assessment', {})
        if risk_assessment.get('overall_risk_level') == 'High':
            immediate_recommendations.append("Address high-risk factors identified in risk assessment")
        
        # Display immediate recommendations
        for i, rec in enumerate(immediate_recommendations, 1):
            html += f'<div class="action-item priority-high">{i}. {rec}</div>'
        
        html += """
        <h3>Medium-term Strategy (1-3 months)</h3>
        """
        
        # Medium-term recommendations
        competitive_threats = predictions.get('competitive_threats', [])
        if competitive_threats:
            medium_term_recommendations.append(f"Develop defensive strategies against {len(competitive_threats)} identified competitive threats")
        
        market_share = results.get('competitive', {}).get('market_share', {}).get('lenovo', 0)
        if market_share < 40:
            medium_term_recommendations.append("Launch market share expansion initiative targeting competitor weaknesses")
        
        for i, rec in enumerate(medium_term_recommendations, 1):
            html += f'<div class="action-item priority-medium">{i}. {rec}</div>'
        
        html += """
        <h3>Long-term Vision (3-12 months)</h3>
        """
        
        # Long-term recommendations
        advanced_metrics = results.get('advanced_metrics', {})
        competitive_status = advanced_metrics.get('market_dominance', {}).get('lenovo', {}).get('competitive_status', '')
        if competitive_status in ['Challenger', 'Strong']:
            long_term_recommendations.append("Strategic initiative to achieve market leadership position")
        
        for i, rec in enumerate(long_term_recommendations, 1):
            html += f'<div class="action-item priority-low">{i}. {rec}</div>'
        
        html += """
        <h3>Success Metrics & KPIs</h3>
        <div class="metric">
            <h4>📊 Key Performance Indicators to Track</h4>
            <ul>
                <li><strong>Traffic Growth:</strong> Month-over-month organic traffic increase</li>
                <li><strong>Position Improvements:</strong> Keywords moving from page 2 to page 1</li>
                <li><strong>Market Share:</strong> Relative competitive position</li>
                <li><strong>SERP Features:</strong> Featured snippet and rich result captures</li>
                <li><strong>Risk Mitigation:</strong> Reduction in identified risk factors</li>
                <li><strong>Competitive Gaps:</strong> Closure of position gaps vs competitors</li>
            </ul>
        </div>
        </div>
        """
        
        return html
    
    # =========================================================================
    # ENHANCED EXISTING METHODS
    # =========================================================================
    
    def _generate_json_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive JSON report"""
        
        enhanced_results = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_period': 'May 19-21, 2025',
                'report_version': '3.0',
                'features': [
                    'core_analysis', 'advanced_keyword_analysis', 'competitive_analysis',
                    'gap_analysis', 'predictive_analytics', 'strategic_planning',
                    'comprehensive_visualizations', 'data_quality_validation'
                ],
                'analysis_completeness': self._calculate_completeness(results)
            },
            'analysis_results': results,
            'summary_statistics': self._generate_summary_statistics(results),
            'key_insights': self._extract_key_insights(results)
        }
        
        report_path = self.reports_dir / "comprehensive_seo_analysis_data.json"
        
        with open(report_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        return str(report_path)
    
    def _generate_excel_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive Excel report with all data"""
        
        report_path = self.reports_dir / "comprehensive_seo_analysis_data.xlsx"
        
        try:
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                # Executive Summary
                exec_summary = self._create_excel_summary(results)
                exec_summary.to_excel(writer, sheet_name='Executive_Summary', index=False)
                
                # Core Analysis Data
                summary_data = results.get('summary', {})
                if summary_data:
                    summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
                    summary_df.to_excel(writer, sheet_name='Performance_Summary', index=True)
                
                # Competitive Analysis
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
                
                # Advanced Analysis
                advanced_data = results.get('advanced_keyword_analysis', {})
                if advanced_data:
                    # Intent distribution
                    intent_df = pd.DataFrame(list(advanced_data.get('intent_distribution', {}).items()),
                                           columns=['Intent_Type', 'Keyword_Count'])
                    intent_df.to_excel(writer, sheet_name='Intent_Distribution', index=False)
                    
                    # SERP features
                    serp_features = advanced_data.get('serp_features', {}).get('feature_breakdown', {})
                    if serp_features:
                        serp_df = pd.DataFrame(list(serp_features.items()),
                                             columns=['SERP_Feature', 'Keyword_Count'])
                        serp_df.to_excel(writer, sheet_name='SERP_Features', index=False)
                
                # Gap Analysis
                gap_data = results.get('competitive_gaps', {})
                if gap_data:
                    # Position gaps
                    position_gaps = gap_data.get('position_gaps', [])
                    if position_gaps:
                        gaps_df = pd.DataFrame(position_gaps)
                        gaps_df.to_excel(writer, sheet_name='Position_Gaps', index=False)
                    
                    # Quick wins
                    quick_wins = gap_data.get('quick_wins', [])
                    if quick_wins:
                        wins_df = pd.DataFrame(quick_wins)
                        wins_df.to_excel(writer, sheet_name='Quick_Wins', index=False)
                
                # Predictive Analytics
                predictions = results.get('predictions', {})
                if predictions:
                    pred_summary = self._create_predictions_summary(predictions)
                    pred_summary.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Strategic Actions
                action_plan = results.get('action_plan', {})
                if action_plan:
                    prioritized_actions = action_plan.get('prioritized_actions', [])
                    if prioritized_actions:
                        actions_df = pd.DataFrame(prioritized_actions)
                        actions_df.to_excel(writer, sheet_name='Strategic_Actions', index=False)
                
                # Data Quality
                validation_data = results.get('data_validation', {})
                if validation_data:
                    quality_summary = self._create_quality_summary(validation_data)
                    quality_summary.to_excel(writer, sheet_name='Data_Quality', index=False)
                
                # Metadata
                metadata_df = pd.DataFrame([{
                    'Generated_At': datetime.now(),
                    'Analysis_Period': 'May 19-21, 2025',
                    'Report_Version': '3.0',
                    'Total_Sheets': len(writer.sheets),
                    'Analysis_Completeness': self._calculate_completeness(results)
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
    
    def _calculate_completeness(self, results: Dict[str, Any]) -> float:
        """Calculate analysis completeness score"""
        
        expected_components = [
            'summary', 'competitive', 'keyword_analysis', 'opportunity_analysis',
            'advanced_keyword_analysis', 'competitive_gaps', 'advanced_metrics',
            'predictions', 'action_plan', 'data_validation'
        ]
        
        completed = len([comp for comp in expected_components if comp in results and results[comp]])
        return completed / len(expected_components)
    
    def _generate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for JSON report"""
        
        summary = results.get('summary', {})
        lenovo_data = summary.get('lenovo', {})
        
        return {
            'total_keywords_analyzed': lenovo_data.get('total_keywords', 0),
            'average_position': lenovo_data.get('avg_position', 0),
            'total_traffic_share': lenovo_data.get('total_traffic', 0),
            'market_share': results.get('competitive', {}).get('market_share', {}).get('lenovo', 0),
            'threat_level': results.get('advanced_metrics', {}).get('threat_assessment', {}).get('threat_level', 'Unknown'),
            'quick_wins_identified': len(results.get('competitive_gaps', {}).get('quick_wins', [])),
            'strategic_actions': len(results.get('action_plan', {}).get('prioritized_actions', [])),
            'data_quality_score': results.get('data_validation', {}).get('summary', {}).get('average_quality_score', 0)
        }
    
    def _extract_key_insights(self, results: Dict[str, Any]) -> List[str]:
        """Extract key insights for JSON report"""
        
        insights = []
        
        # Performance insights
        summary = results.get('summary', {})
        lenovo_data = summary.get('lenovo', {})
        
        if lenovo_data.get('avg_position', 0) < 10:
            insights.append("Strong average position performance - majority of keywords ranking well")
        
        # Market share insights
        market_share = results.get('competitive', {}).get('market_share', {}).get('lenovo', 0)
        if market_share > 40:
            insights.append("Dominant market position with significant competitive advantage")
        elif market_share < 25:
            insights.append("Significant market share growth opportunity available")
        
        # Threat insights
        threat_level = results.get('advanced_metrics', {}).get('threat_assessment', {}).get('threat_level', 'Unknown')
        if threat_level == 'High':
            insights.append("High competitive threat level requires immediate defensive action")
        elif threat_level == 'Medium':
            insights.append("Moderate competitive threats detected - monitor closely")
        
        # Gap analysis insights
        gap_data = results.get('competitive_gaps', {})
        position_gaps = gap_data.get('position_gaps', [])
        quick_wins = gap_data.get('quick_wins', [])
        
        if len(position_gaps) > 50:
            insights.append(f"Significant position gaps identified: {len(position_gaps)} keywords behind competitors")
        
        if len(quick_wins) > 20:
            insights.append(f"Multiple quick win opportunities available: {len(quick_wins)} high-potential keywords")
        
        # Advanced keyword insights
        advanced_data = results.get('advanced_keyword_analysis', {})
        if advanced_data:
            intent_dist = advanced_data.get('intent_distribution', {})
            commercial_ratio = intent_dist.get('commercial', 0) / sum(intent_dist.values()) if intent_dist else 0
            
            if commercial_ratio > 0.3:
                insights.append("High commercial intent keyword focus - good for conversions")
            elif commercial_ratio < 0.1:
                insights.append("Low commercial intent coverage - opportunity for revenue growth")
            
            serp_coverage = advanced_data.get('serp_features', {}).get('coverage_percentage', 0)
            if serp_coverage < 30:
                insights.append("Low SERP features coverage - significant optimization opportunity")
        
        # Predictive insights
        predictions = results.get('predictions', {})
        if predictions:
            traffic_forecast = predictions.get('traffic_forecast', {})
            forecast_change = traffic_forecast.get('forecast_change', 0)
            
            if forecast_change > 10:
                insights.append("Positive traffic growth trajectory predicted")
            elif forecast_change < -10:
                insights.append("Declining traffic trend requires strategic intervention")
            
            risk_level = predictions.get('risk_assessment', {}).get('overall_risk_level', 'Unknown')
            if risk_level == 'High':
                insights.append("High business risk level from SEO vulnerabilities")
        
        return insights

    def _create_excel_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create executive summary for Excel export"""
        
        summary = results.get('summary', {})
        competitive = results.get('competitive', {})
        advanced_metrics = results.get('advanced_metrics', {})
        predictions = results.get('predictions', {})
        
        # Create summary data
        summary_data = []
        
        # Performance metrics
        lenovo_data = summary.get('lenovo', {})
        if lenovo_data:
            summary_data.append({
                'Metric': 'Total Keywords',
                'Value': lenovo_data.get('total_keywords', 0),
                'Category': 'Performance'
            })
            summary_data.append({
                'Metric': 'Average Position',
                'Value': round(lenovo_data.get('avg_position', 0), 1),
                'Category': 'Performance'
            })
            summary_data.append({
                'Metric': 'Traffic Share (%)',
                'Value': round(lenovo_data.get('total_traffic', 0), 1),
                'Category': 'Performance'
            })
            summary_data.append({
                'Metric': 'Top 10 Rankings',
                'Value': lenovo_data.get('top_10_count', 0),
                'Category': 'Performance'
            })
        
        # Competitive metrics
        market_share = competitive.get('market_share', {})
        if market_share:
            summary_data.append({
                'Metric': 'Market Share (%)',
                'Value': round(market_share.get('lenovo', 0), 1),
                'Category': 'Competitive'
            })
        
        # Advanced metrics
        if advanced_metrics:
            threat_level = advanced_metrics.get('threat_assessment', {}).get('threat_level', 'Unknown')
            summary_data.append({
                'Metric': 'Threat Level',
                'Value': threat_level,
                'Category': 'Advanced'
            })
            
            hhi = advanced_metrics.get('lenovo_hhi', 0)
            summary_data.append({
                'Metric': 'Traffic Concentration (HHI)',
                'Value': round(hhi, 3),
                'Category': 'Advanced'
            })
        
        # Predictions
        if predictions:
            forecast_change = predictions.get('traffic_forecast', {}).get('forecast_change', 0)
            summary_data.append({
                'Metric': '30-Day Traffic Forecast (%)',
                'Value': round(forecast_change, 1),
                'Category': 'Predictive'
            })
            
            risk_level = predictions.get('risk_assessment', {}).get('overall_risk_level', 'Unknown')
            summary_data.append({
                'Metric': 'Overall Risk Level',
                'Value': risk_level,
                'Category': 'Predictive'
            })
        
        return pd.DataFrame(summary_data)

    def _create_predictions_summary(self, predictions: Dict[str, Any]) -> pd.DataFrame:
        """Create predictions summary for Excel export"""
        
        pred_data = []
        
        # Traffic forecast
        traffic_forecast = predictions.get('traffic_forecast', {})
        if traffic_forecast:
            pred_data.append({
                'Prediction_Type': 'Traffic Forecast',
                'Current_Trend': traffic_forecast.get('current_trend', 'Unknown'),
                'Forecast_Change': traffic_forecast.get('forecast_change', 0),
                'Reliability': traffic_forecast.get('forecast_reliability', 'Unknown'),
                'Timeline': '30 days'
            })
        
        # Market share trajectory
        market_trajectory = predictions.get('market_share_trajectory', {})
        if market_trajectory:
            pred_data.append({
                'Prediction_Type': 'Market Share',
                'Current_Value': market_trajectory.get('current_share', 0),
                'Projected_Change': market_trajectory.get('projected_3_month_change', 0),
                'Projected_Value': market_trajectory.get('projected_share', 0),
                'Timeline': '3 months'
            })
        
        # Competitive threats
        competitive_threats = predictions.get('competitive_threats', [])
        for threat in competitive_threats:
            pred_data.append({
                'Prediction_Type': 'Competitive Threat',
                'Competitor': threat.get('competitor', 'Unknown'),
                'Threat_Level': threat.get('threat_level', 'Unknown'),
                'Impact': threat.get('predicted_impact', 'Unknown'),
                'Timeline': threat.get('timeline', 'Unknown')
            })
        
        # Risk assessment
        risk_assessment = predictions.get('risk_assessment', {})
        if risk_assessment:
            pred_data.append({
                'Prediction_Type': 'Risk Assessment',
                'Risk_Level': risk_assessment.get('overall_risk_level', 'Unknown'),
                'Risk_Factors': len(risk_assessment.get('identified_risks', [])),
                'Priority_Areas': '; '.join(risk_assessment.get('priority_mitigations', [])[:3])
            })
        
        return pd.DataFrame(pred_data)

    def _create_quality_summary(self, validation_data: Dict[str, Any]) -> pd.DataFrame:
        """Create data quality summary for Excel export"""
        
        quality_data = []
        
        # Overall summary
        summary = validation_data.get('summary', {})
        if summary:
            quality_data.append({
                'Dataset': 'Overall',
                'Status': summary.get('overall_status', 'Unknown'),
                'Quality_Score': round(summary.get('average_quality_score', 0), 2),
                'Total_Rows': summary.get('total_rows', 0),
                'Issues_Count': 0
            })
        
        # Individual datasets
        dataset_results = {k: v for k, v in validation_data.items() if k != 'summary'}
        for dataset, data in dataset_results.items():
            quality_data.append({
                'Dataset': dataset.title(),
                'Status': data.get('status', 'Unknown'),
                'Quality_Score': round(data.get('quality_score', 0), 2),
                'Row_Count': data.get('row_count', 0),
                'Issues_Count': len(data.get('issues', [])),
                'Completeness': round(data.get('completeness', 0), 2),
                'Duplicates': data.get('duplicate_info', {}).get('count', 0)
            })
        
        return pd.DataFrame(quality_data)

    # =========================================================================
    # MISSING ADVANCED VISUALIZATION METHODS
    # =========================================================================

    def _create_market_dominance_chart(self, market_dominance: Dict[str, Any], viz_dir: Path) -> str:
        """Create market dominance bubble chart"""
        
        try:
            plt.figure(figsize=(12, 8))
            
            companies = []
            market_shares = []
            dominance_scores = []
            colors = ['#2E86C1', '#E74C3C', '#27AE60']
            
            for i, (company, data) in enumerate(market_dominance.items()):
                companies.append(company.title())
                market_shares.append(data.get('market_share', 0) * 100)  # Convert to percentage
                dominance_scores.append(data.get('dominance_score', 0) * 1000)  # Scale for bubble size
            
            # Create bubble chart
            scatter = plt.scatter(range(len(companies)), market_shares, 
                                s=dominance_scores, alpha=0.7, c=colors[:len(companies)])
            
            # Customize chart
            plt.xticks(range(len(companies)), companies)
            plt.ylabel('Market Share (%)', fontsize=12, fontweight='bold')
            plt.title('Market Dominance Analysis\n(Bubble size = Dominance Score)', fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels
            for i, (company, share, score) in enumerate(zip(companies, market_shares, dominance_scores)):
                plt.annotate(f'{share:.1f}%\nScore: {score/1000:.3f}', 
                        (i, share), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            chart_path = viz_dir / "market_dominance_bubble.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating market dominance chart: {e}")
            return ""

    def _create_risk_dashboard_chart(self, risk_assessment: Dict[str, Any], viz_dir: Path) -> str:
        """Create risk assessment dashboard"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Risk level gauge
            risk_level = risk_assessment.get('overall_risk_level', 'Unknown')
            risk_factors = risk_assessment.get('identified_risks', [])
            
            # Risk level chart
            levels = ['Low', 'Medium', 'High']
            risk_colors = ['#2ECC71', '#F39C12', '#E74C3C']
            current_index = levels.index(risk_level) if risk_level in levels else 1
            
            bars1 = ax1.bar(levels, [1, 1, 1], color=['lightgray' if i != current_index else risk_colors[i] for i in range(3)])
            bars1[current_index].set_edgecolor('black')
            bars1[current_index].set_linewidth(3)
            ax1.set_title(f'Overall Risk Level: {risk_level}', fontweight='bold')
            ax1.set_ylabel('Risk Assessment')
            
            # Risk factors breakdown
            if risk_factors:
                risk_types = [risk['risk_type'] for risk in risk_factors]
                risk_severities = [risk['severity'] for risk in risk_factors]
                
                severity_counts = {'High': 0, 'Medium': 0, 'Low': 0}
                for severity in risk_severities:
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                
                ax2.pie(severity_counts.values(), labels=severity_counts.keys(), autopct='%1.0f',
                    colors=['#E74C3C', '#F39C12', '#2ECC71'])
                ax2.set_title('Risk Factors by Severity', fontweight='bold')
            
            # Risk categories
            if risk_factors:
                risk_type_counts = {}
                for risk in risk_factors:
                    risk_type = risk['risk_type']
                    risk_type_counts[risk_type] = risk_type_counts.get(risk_type, 0) + 1
                
                if risk_type_counts:
                    ax3.barh(list(risk_type_counts.keys()), list(risk_type_counts.values()), color='coral')
                    ax3.set_title('Risk Categories', fontweight='bold')
                    ax3.set_xlabel('Number of Risk Factors')
            
            # Risk timeline/priority
            if risk_factors:
                priorities = [risk.get('severity', 'Medium') for risk in risk_factors]
                priority_timeline = {'Immediate': 0, 'Short-term': 0, 'Long-term': 0}
                
                for priority in priorities:
                    if priority == 'High':
                        priority_timeline['Immediate'] += 1
                    elif priority == 'Medium':
                        priority_timeline['Short-term'] += 1
                    else:
                        priority_timeline['Long-term'] += 1
                
                ax4.bar(priority_timeline.keys(), priority_timeline.values(), 
                    color=['#E74C3C', '#F39C12', '#2ECC71'])
                ax4.set_title('Risk Priority Timeline', fontweight='bold')
                ax4.set_ylabel('Number of Risks')
            
            plt.suptitle('Risk Assessment Dashboard', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = viz_dir / "risk_dashboard.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating risk dashboard chart: {e}")
            return ""

    # =========================================================================
    # COMPREHENSIVE HTML REPORT SECTIONS (Continued)
    # =========================================================================

    def _generate_comprehensive_html_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report with ALL analysis"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive SEO Competitive Intelligence Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f8f9fa; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ background: white; margin: 20px 0; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ background: #e8f5e8; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #28a745; }}
                .competitor {{ background: #fff3cd; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #ffc107; }}
                .insight {{ background: #d1ecf1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #17a2b8; }}
                .warning {{ background: #f8d7da; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #dc3545; }}
                .success {{ background: #d4edda; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #28a745; }}
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
                .two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .action-item {{ background: #e7f3ff; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #007bff; }}
                .priority-high {{ border-left-color: #dc3545; background: #f8d7da; }}
                .priority-medium {{ border-left-color: #ffc107; background: #fff3cd; }}
                .priority-low {{ border-left-color: #28a745; background: #d4edda; }}
                .nav-menu {{ background: #343a40; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .nav-menu a {{ color: white; text-decoration: none; margin: 0 15px; }}
                .nav-menu a:hover {{ color: #ffc107; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Comprehensive SEO Competitive Intelligence Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Analysis Period:</strong> May 19-21, 2025</p>
                <p><strong>Features:</strong> Complete Analysis • Advanced Metrics • Predictive Analytics • Strategic Planning</p>
            </div>
            
            <div class="nav-menu">
                <a href="#executive-summary">Executive Summary</a>
                <a href="#data-quality">Data Quality</a>
                <a href="#core-analysis">Core Analysis</a>
                <a href="#advanced-analysis">Advanced Analysis</a>
                <a href="#competitive-analysis">Competitive Analysis</a>
                <a href="#gap-analysis">Gap Analysis</a>
                <a href="#predictive-analytics">Predictive Analytics</a>
                <a href="#strategic-actions">Strategic Actions</a>
            </div>
            
            {self._create_comprehensive_executive_summary_html(results)}
            {self._create_data_quality_section_comprehensive(results.get('data_validation', {}))}
            {self._create_core_analysis_section_comprehensive(results)}
            {self._create_advanced_analysis_section_comprehensive(results.get('advanced_keyword_analysis', {}))}
            {self._create_competitive_analysis_section_comprehensive(results.get('competitive', {}), results.get('advanced_metrics', {}))}
            {self._create_gap_analysis_section_comprehensive(results.get('competitive_gaps', {}))}
            {self._create_predictive_analytics_section_comprehensive(results.get('predictions', {}), results.get('advanced_metrics', {}))}
            {self._create_strategic_actions_section_comprehensive(results.get('action_plan', {}))}
            {self._create_final_recommendations_section(results)}
        </body>
        </html>
        """
        
        report_path = self.reports_dir / "comprehensive_seo_intelligence_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(report_path)

    def _create_comprehensive_executive_summary_html(self, results: Dict[str, Any]) -> str:
        """Create comprehensive executive summary with all metrics"""
        
        summary = results.get('summary', {})
        competitive = results.get('competitive', {})
        advanced_metrics = results.get('advanced_metrics', {})
        predictions = results.get('predictions', {})
        
        lenovo_data = summary.get('lenovo', {})
        market_share = competitive.get('market_share', {})
        
        html = f"""
        <div class="section" id="executive-summary">
            <h2>📈 Executive Summary</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{lenovo_data.get('total_keywords', 0):,}</div>
                    <div class="stat-label">Total Keywords Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{lenovo_data.get('avg_position', 0):.1f}</div>
                    <div class="stat-label">Average Position</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{lenovo_data.get('total_traffic', 0):.1f}%</div>
                    <div class="stat-label">Total Traffic Share</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{market_share.get('lenovo', 0):.1f}%</div>
                    <div class="stat-label">Market Share</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{lenovo_data.get('top_10_count', 0):,}</div>
                    <div class="stat-label">Top 10 Rankings</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{advanced_metrics.get('threat_assessment', {}).get('threat_level', 'Unknown')}</div>
                    <div class="stat-label">Threat Level</div>
                </div>
            </div>
            
            <div class="two-column">
                <div>
                    <h3>Key Performance Indicators</h3>
                    <div class="metric">
                        <h4>🎯 Position Performance</h4>
                        <p><strong>Top 3 Rankings:</strong> {lenovo_data.get('top_3_count', 0):,} keywords</p>
                        <p><strong>Page 1 Rankings:</strong> {lenovo_data.get('page_1_count', 0):,} keywords</p>
                        <p><strong>Page 2 Rankings:</strong> {lenovo_data.get('page_2_count', 0):,} keywords</p>
                    </div>
                </div>
                <div>
                    <h3>Strategic Status</h3>
                    <div class="metric">
                        <h4>🏆 Competitive Position</h4>
                        <p><strong>Status:</strong> {advanced_metrics.get('market_dominance', {}).get('lenovo', {}).get('competitive_status', 'Unknown')}</p>
                        <p><strong>Market Share:</strong> {market_share.get('lenovo', 0):.1f}%</p>
                        <p><strong>Traffic Risk:</strong> {advanced_metrics.get('traffic_concentration_risk', 'Unknown')}</p>
                    </div>
                </div>
            </div>
        """
        
        # Add predictive insights if available
        if predictions:
            forecast_change = predictions.get('traffic_forecast', {}).get('forecast_change', 0)
            risk_level = predictions.get('risk_assessment', {}).get('overall_risk_level', 'Unknown')
            
            html += f"""
            <div class="two-column">
                <div>
                    <h3>Predictive Insights</h3>
                    <div class="{'success' if forecast_change > 0 else 'warning' if forecast_change < 0 else 'insight'}">
                        <h4>📈 30-Day Forecast</h4>
                        <p><strong>Traffic Change:</strong> {forecast_change:+.1f}%</p>
                        <p><strong>Reliability:</strong> {predictions.get('traffic_forecast', {}).get('forecast_reliability', 'Unknown')}</p>
                    </div>
                </div>
                <div>
                    <h3>Risk Assessment</h3>
                    <div class="{'warning' if risk_level == 'High' else 'insight'}">
                        <h4>🛡️ Overall Risk Level: {risk_level}</h4>
                        <p><strong>Risk Factors:</strong> {len(predictions.get('risk_assessment', {}).get('identified_risks', []))}</p>
                        <p><strong>Active Threats:</strong> {len(predictions.get('competitive_threats', []))}</p>
                    </div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html

    def _create_final_recommendations_section(self, results: Dict[str, Any]) -> str:
        """Create final comprehensive recommendations section"""
        
        html = f"""
        <div class="section">
            <h2>🎯 Strategic Implementation Roadmap</h2>
            
            <h3>Executive Action Items</h3>
            <div class="action-item priority-high">
                <h4>🚨 Immediate Priority (0-30 days)</h4>
                <ul>
                    <li>Address {len(results.get('competitive_gaps', {}).get('quick_wins', []))} quick win opportunities</li>
                    <li>Implement risk mitigation for {results.get('advanced_metrics', {}).get('threat_assessment', {}).get('threat_level', 'Unknown')} threat level</li>
                    <li>Optimize SERP features coverage (currently {results.get('advanced_keyword_analysis', {}).get('serp_features', {}).get('coverage_percentage', 0):.1f}%)</li>
                </ul>
            </div>
            
            <div class="action-item priority-medium">
                <h4>📊 Strategic Focus (1-3 months)</h4>
                <ul>
                    <li>Close {len(results.get('competitive_gaps', {}).get('position_gaps', []))} position gaps vs competitors</li>
                    <li>Expand market share from {results.get('competitive', {}).get('market_share', {}).get('lenovo', 0):.1f}%</li>
                    <li>Diversify keyword portfolio to reduce traffic concentration risk</li>
                </ul>
            </div>
            
            <div class="action-item priority-low">
                <h4>🏗️ Long-term Vision (3-12 months)</h4>
                <ul>
                    <li>Achieve market leadership position</li>
                    <li>Build predictive SEO monitoring system</li>
                    <li>Establish competitive early warning systems</li>
                </ul>
            </div>
            
            <h3>Success Metrics & KPIs</h3>
            <div class="metric">
                <h4>📊 Track These Key Indicators</h4>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">+15%</div>
                        <div class="stat-label">Target Traffic Growth</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">+5%</div>
                        <div class="stat-label">Market Share Goal</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">-50%</div>
                        <div class="stat-label">Position Gap Reduction</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">Low</div>
                        <div class="stat-label">Target Risk Level</div>
                    </div>
                </div>
            </div>
            
            <h3>Resource Allocation</h3>
            <div class="two-column">
                <div class="metric">
                    <h4>💰 Budget Priority</h4>
                    <ul>
                        <li>60% - Quick wins & gap closure</li>
                        <li>25% - Competitive defense</li>
                        <li>15% - Innovation & expansion</li>
                    </ul>
                </div>
                <div class="metric">
                    <h4>👥 Team Focus</h4>
                    <ul>
                        <li>Technical SEO - Position optimization</li>
                        <li>Content Strategy - SERP features</li>
                        <li>Analytics - Competitive monitoring</li>
                    </ul>
                </div>
            </div>
        </div>
        """
        
        return html

    def _create_data_quality_section_comprehensive(self, validation_data: Dict[str, Any]) -> str:
        """Create comprehensive data quality section for HTML report"""
        
        if not validation_data:
            return ""
        
        summary = validation_data.get('summary', {})
        quality_score = summary.get('average_quality_score', 0)
        
        quality_color = 'success' if quality_score > 0.8 else 'warning' if quality_score > 0.6 else 'danger'
        
        html = f"""
        <div class="section" id="data-quality">
            <h2>📊 Data Quality Assessment</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" style="color: {'#28a745' if quality_score > 0.8 else '#ffc107' if quality_score > 0.6 else '#dc3545'}">{quality_score:.2f}</div>
                    <div class="stat-label">Overall Quality Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('total_datasets', 0)}</div>
                    <div class="stat-label">Datasets Validated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('total_rows', 0):,}</div>
                    <div class="stat-label">Total Records</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.get('overall_status', 'Unknown').title()}</div>
                    <div class="stat-label">Quality Status</div>
                </div>
            </div>
        """
        
        # Add dataset-specific quality metrics
        dataset_results = {k: v for k, v in validation_data.items() if k != 'summary'}
        if dataset_results:
            html += """
            <h3>Dataset Quality Breakdown</h3>
            <table>
                <tr><th>Dataset</th><th>Status</th><th>Rows</th><th>Quality Score</th><th>Issues</th></tr>
            """
            
            for dataset, data in dataset_results.items():
                status_color = 'highlight' if data.get('status') in ['excellent', 'good'] else ''
                html += f"""
                <tr class="{status_color}">
                    <td><strong>{dataset.title()}</strong></td>
                    <td>{data.get('status', 'Unknown').title()}</td>
                    <td>{data.get('row_count', 0):,}</td>
                    <td>{data.get('quality_score', 0):.2f}</td>
                    <td>{len(data.get('issues', []))}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            html += "<h3>Quality Improvement Recommendations</h3>"
            for rec in recommendations:
                html += f'<div class="insight">💡 {rec}</div>'
        
        html += "</div>"
        return html

    def _create_core_analysis_section_comprehensive(self, results: Dict[str, Any]) -> str:
        """Create comprehensive core analysis section"""
        
        summary = results.get('summary', {})
        keyword_analysis = results.get('keyword_analysis', {})
        opportunity_analysis = results.get('opportunity_analysis', {})
        
        html = f"""
        <div class="section" id="core-analysis">
            <h2>🔍 Core SEO Analysis</h2>
            
            <div class="two-column">
                <div>
                    <h3>Performance Overview</h3>
        """
        
        # Add performance metrics for each company
        for company, data in summary.items():
            if company == 'gap_keywords':
                continue
            
            html += f"""
            <div class="metric">
                <h4>📊 {company.title()} Performance</h4>
                <p><strong>Keywords:</strong> {data.get('total_keywords', 0):,}</p>
                <p><strong>Avg Position:</strong> {data.get('avg_position', 0):.1f}</p>
                <p><strong>Traffic:</strong> {data.get('total_traffic', 0):.1f}%</p>
                <p><strong>Top 10:</strong> {data.get('top_10_count', 0):,}</p>
            </div>
            """
        
        html += """
                </div>
                <div>
                    <h3>Position Analysis</h3>
        """
        
        # Add position distribution if available
        position_dist = keyword_analysis.get('position_distribution', {})
        if position_dist:
            html += """
            <div class="metric">
                <h4>🎯 Position Distribution</h4>
            """
            for position_range, count in position_dist.items():
                html += f"<p><strong>{position_range.replace('_', ' ').title()}:</strong> {count:,} keywords</p>"
            
            html += "</div>"
        
        html += """
                </div>
            </div>
        """
        
        # Add top performing keywords
        best_keywords = keyword_analysis.get('best_keywords', [])
        if best_keywords:
            html += """
            <h3>🏆 Top Performing Keywords</h3>
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
        
        # Add opportunity overview
        if opportunity_analysis:
            total_gaps = opportunity_analysis.get('total_gap_keywords', 0)
            opportunities = opportunity_analysis.get('high_value_opportunities', [])
            
            html += f"""
            <h3>💡 Opportunity Overview</h3>
            <div class="metric">
                <h4>🎯 Gap Analysis Summary</h4>
                <p><strong>Total Gap Keywords:</strong> {total_gaps:,}</p>
                <p><strong>High-Value Opportunities:</strong> {len(opportunities)}</p>
                <p><strong>Avg Difficulty:</strong> {opportunity_analysis.get('avg_difficulty', 0):.1f}</p>
            </div>
            """
        
        html += "</div>"
        return html

    def _create_advanced_analysis_section_comprehensive(self, advanced_data: Dict[str, Any]) -> str:
        """Create comprehensive advanced analysis section"""
        
        if not advanced_data:
            return ""
        
        html = f"""
        <div class="section" id="advanced-analysis">
            <h2>🎯 Advanced Keyword Analysis</h2>
        """
        
        # Intent distribution
        intent_dist = advanced_data.get('intent_distribution', {})
        if intent_dist:
            html += """
            <h3>Search Intent Distribution</h3>
            <div class="stats-grid">
            """
            for intent, count in intent_dist.items():
                html += f"""
                <div class="stat-card">
                    <div class="stat-value">{count:,}</div>
                    <div class="stat-label">{intent.title()} Keywords</div>
                </div>
                """
            html += "</div>"
        
        # SERP features analysis
        serp_features = advanced_data.get('serp_features', {})
        if serp_features:
            coverage = serp_features.get('coverage_percentage', 0)
            total_features = serp_features.get('total_with_features', 0)
            
            html += f"""
            <div class="two-column">
                <div>
                    <h3>SERP Features Analysis</h3>
                    <div class="metric">
                        <h4>🔍 Coverage Overview</h4>
                        <p><strong>Coverage:</strong> {coverage:.1f}%</p>
                        <p><strong>Keywords with Features:</strong> {total_features:,}</p>
                    </div>
                </div>
                <div>
                    <h3>Top Features</h3>
                    <div class="metric">
                        <h4>📊 Feature Breakdown</h4>
            """
            
            top_features = serp_features.get('top_features', [])
            for feature, count in top_features[:5]:
                html += f"<p><strong>{feature}:</strong> {count} keywords</p>"
            
            html += """
                    </div>
                </div>
            </div>
            """
        
        # Branded vs non-branded analysis
        branded_analysis = advanced_data.get('branded_vs_nonbranded', {})
        if branded_analysis:
            html += f"""
            <h3>Branded vs Non-Branded Performance</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{branded_analysis.get('branded_count', 0):,}</div>
                    <div class="stat-label">Branded Keywords</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{branded_analysis.get('non_branded_count', 0):,}</div>
                    <div class="stat-label">Non-Branded Keywords</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{branded_analysis.get('branded_avg_position', 0):.1f}</div>
                    <div class="stat-label">Branded Avg Position</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{branded_analysis.get('non_branded_avg_position', 0):.1f}</div>
                    <div class="stat-label">Non-Branded Avg Position</div>
                </div>
            </div>
            """
            
            # Add insights
            insights = branded_analysis.get('insights', [])
            if insights:
                for insight in insights:
                    html += f'<div class="insight">💡 {insight}</div>'
        
        html += "</div>"
        return html

    def _create_competitive_analysis_section_comprehensive(self, competitive_data: Dict[str, Any], advanced_metrics: Dict[str, Any]) -> str:
        """Create comprehensive competitive analysis section"""
        
        html = f"""
        <div class="section" id="competitive-analysis">
            <h2>🏆 Comprehensive Competitive Analysis</h2>
        """
        
        # Market share and traffic comparison
        market_share = competitive_data.get('market_share', {})
        traffic_comparison = competitive_data.get('traffic_comparison', {})
        
        if market_share:
            html += """
            <h3>Market Share Analysis</h3>
            <div class="stats-grid">
            """
            for company, share in market_share.items():
                html += f"""
                <div class="stat-card">
                    <div class="stat-value">{share:.1f}%</div>
                    <div class="stat-label">{company.title()} Market Share</div>
                </div>
                """
            html += "</div>"
        
        # Advanced competitive metrics
        if advanced_metrics:
            threat_assessment = advanced_metrics.get('threat_assessment', {})
            market_dominance = advanced_metrics.get('market_dominance', {})
            
            if threat_assessment:
                threat_level = threat_assessment.get('threat_level', 'Unknown')
                active_threats = threat_assessment.get('active_threats', [])
                
                html += f"""
                <h3>Threat Assessment</h3>
                <div class="{'warning' if threat_level == 'High' else 'insight'}">
                    <h4>🚨 Current Threat Level: {threat_level}</h4>
                    <p><strong>Active Threats:</strong> {len(active_threats)}</p>
                """
                
                if active_threats:
                    html += "<p><strong>Threat Details:</strong></p><ul>"
                    for threat in active_threats[:3]:
                        html += f"<li>{threat}</li>"
                    html += "</ul>"
                
                html += "</div>"
            
            if market_dominance:
                html += """
                <h3>Market Dominance Analysis</h3>
                <table>
                    <tr><th>Company</th><th>Market Share</th><th>Dominance Score</th><th>Status</th></tr>
                """
                
                for company, data in market_dominance.items():
                    status = data.get('competitive_status', 'Unknown')
                    share = data.get('market_share', 0) * 100
                    score = data.get('dominance_score', 0)
                    
                    row_class = 'highlight' if company == 'lenovo' else ''
                    html += f"""
                    <tr class="{row_class}">
                        <td><strong>{company.title()}</strong></td>
                        <td>{share:.1f}%</td>
                        <td>{score:.3f}</td>
                        <td>{status}</td>
                    </tr>
                    """
                
                html += "</table>"
        
        # Keyword overlap analysis
        keyword_overlap = competitive_data.get('keyword_overlap', {})
        if keyword_overlap:
            html += """
            <h3>Keyword Overlap Analysis</h3>
            <div class="two-column">
            """
            
            for competitor, overlap in keyword_overlap.items():
                html += f"""
                <div class="competitor">
                    <h4>📊 Shared with {competitor.title()}</h4>
                    <p><strong>Common Keywords:</strong> {overlap:,}</p>
                </div>
                """
            
            html += "</div>"
        
        html += "</div>"
        return html

    def _create_gap_analysis_section_comprehensive(self, gap_data: Dict[str, Any]) -> str:
        """Create comprehensive gap analysis section"""
        
        if not gap_data:
            return ""
        
        html = f"""
        <div class="section" id="gap-analysis">
            <h2>📉 Competitive Gap Analysis</h2>
        """
        
        # Position gaps
        position_gaps = gap_data.get('position_gaps', [])
        quick_wins = gap_data.get('quick_wins', [])
        
        if position_gaps:
            html += f"""
            <h3>Position Gaps ({len(position_gaps)} identified)</h3>
            <table>
                <tr><th>Keyword</th><th>Lenovo Position</th><th>Competitor</th><th>Competitor Position</th><th>Gap</th><th>Priority</th></tr>
            """
            
            for gap in position_gaps[:10]:
                priority_class = 'priority-high' if gap.get('priority') == 'high' else 'priority-medium'
                html += f"""
                <tr>
                    <td>{gap['keyword']}</td>
                    <td>{gap['lenovo_position']}</td>
                    <td>{gap['competitor']}</td>
                    <td>{gap['competitor_position']}</td>
                    <td class="{priority_class}">{gap['gap']}</td>
                    <td>{gap.get('priority', 'medium').title()}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Quick wins
        if quick_wins:
            html += f"""
            <h3>Quick Win Opportunities ({len(quick_wins)} identified)</h3>
            <div class="two-column">
            """
            
            page_2_wins = [qw for qw in quick_wins if qw.get('type') == 'page_2_keywords']
            gap_wins = [qw for qw in quick_wins if qw.get('type') == 'easy_gap_keywords']
            
            if page_2_wins:
                html += """
                <div>
                    <h4>⚡ Page 2 Keywords</h4>
                """
                for win in page_2_wins[:5]:
                    html += f"""
                    <div class="success">
                        <strong>{win['keyword']}</strong><br>
                        Position: {win['current_position']} | Traffic: {win.get('traffic_potential', 0):.1f}%
                    </div>
                    """
                html += "</div>"
            
            if gap_wins:
                html += """
                <div>
                    <h4>🎯 Easy Gap Keywords</h4>
                """
                for win in gap_wins[:5]:
                    html += f"""
                    <div class="success">
                        <strong>{win['keyword']}</strong><br>
                        Volume: {win.get('volume', 0):,} | Difficulty: {win.get('difficulty', 0)}
                    </div>
                    """
                html += "</div>"
            
            html += "</div>"
        
        html += "</div>"
        return html

    def _create_predictive_analytics_section_comprehensive(self, predictions: Dict[str, Any], advanced_metrics: Dict[str, Any]) -> str:
        """Create comprehensive predictive analytics section"""
        
        if not predictions:
            return ""
        
        html = f"""
        <div class="section" id="predictive-analytics">
            <h2>🔮 Predictive Analytics & Forecasting</h2>
        """
        
        # Traffic forecast
        traffic_forecast = predictions.get('traffic_forecast', {})
        if traffic_forecast:
            forecast_change = traffic_forecast.get('forecast_change', 0)
            reliability = traffic_forecast.get('forecast_reliability', 'Unknown')
            
            forecast_class = 'success' if forecast_change > 0 else 'warning' if forecast_change < 0 else 'insight'
            
            html += f"""
            <h3>Traffic Forecast (30-Day Projection)</h3>
            <div class="{forecast_class}">
                <h4>📈 Predicted Change: {forecast_change:+.1f}%</h4>
                <p><strong>Forecast Reliability:</strong> {reliability}</p>
                <p><strong>Current Trend:</strong> {traffic_forecast.get('current_trend', 'Unknown')}</p>
            </div>
            """
        
        # Market share trajectory
        market_trajectory = predictions.get('market_share_trajectory', {})
        if market_trajectory:
            current_share = market_trajectory.get('current_share', 0)
            projected_change = market_trajectory.get('projected_3_month_change', 0)
            projected_share = market_trajectory.get('projected_share', 0)
            
            html += f"""
            <h3>Market Share Trajectory</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{current_share:.1f}%</div>
                    <div class="stat-label">Current Share</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{projected_change:+.1f}%</div>
                    <div class="stat-label">3-Month Change</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{projected_share:.1f}%</div>
                    <div class="stat-label">Projected Share</div>
                </div>
            </div>
            """
        
        # Competitive threats
        competitive_threats = predictions.get('competitive_threats', [])
        if competitive_threats:
            html += f"""
            <h3>Competitive Threat Predictions ({len(competitive_threats)} active)</h3>
            """
            
            for threat in competitive_threats:
                threat_class = 'warning' if threat.get('threat_level') == 'Critical' else 'insight'
                html += f"""
                <div class="{threat_class}">
                    <h4>⚠️ {threat['competitor'].title()} - {threat['threat_level']} Threat</h4>
                    <p><strong>Impact:</strong> {threat['predicted_impact']}</p>
                    <p><strong>Timeline:</strong> {threat['timeline']}</p>
                    <p><strong>Recommendation:</strong> {threat['recommendation']}</p>
                </div>
                """
        
        # Risk assessment
        risk_assessment = predictions.get('risk_assessment', {})
        if risk_assessment:
            risk_level = risk_assessment.get('overall_risk_level', 'Unknown')
            identified_risks = risk_assessment.get('identified_risks', [])
            
            risk_class = 'warning' if risk_level == 'High' else 'insight'
            
            html += f"""
            <h3>Risk Assessment</h3>
            <div class="{risk_class}">
                <h4>🛡️ Overall Risk Level: {risk_level}</h4>
                <p><strong>Risk Factors Identified:</strong> {len(identified_risks)}</p>
            """
            
            if identified_risks:
                html += "<h5>Key Risk Factors:</h5><ul>"
                for risk in identified_risks[:3]:
                    html += f"<li><strong>{risk['risk_type']}:</strong> {risk['description']}</li>"
                html += "</ul>"
            
            html += "</div>"
        
        html += "</div>"
        return html

    def _create_strategic_actions_section_comprehensive(self, action_plan: Dict[str, Any]) -> str:
        """Create comprehensive strategic actions section"""
        
        if not action_plan:
            return ""
        
        html = f"""
        <div class="section" id="strategic-actions">
            <h2>📋 Strategic Action Plan</h2>
        """
        
        # Action categories
        immediate_actions = action_plan.get('immediate_actions', [])
        defensive_actions = action_plan.get('defensive_actions', [])
        offensive_actions = action_plan.get('offensive_actions', [])
        
        # Overview
        html += f"""
        <h3>Action Plan Overview</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(immediate_actions)}</div>
                <div class="stat-label">Immediate Actions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(defensive_actions)}</div>
                <div class="stat-label">Defensive Actions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(offensive_actions)}</div>
                <div class="stat-label">Offensive Actions</div>
            </div>
        </div>
        """
        
        # Prioritized actions
        prioritized_actions = action_plan.get('prioritized_actions', [])
        if prioritized_actions:
            html += """
            <h3>Top Priority Actions</h3>
            """
            
            for i, action in enumerate(prioritized_actions[:10], 1):
                priority_score = action.get('priority_score', 0)
                category = action.get('category', 'unknown').replace('_', ' ').title()
                
                if priority_score > 8:
                    priority_class = 'priority-high'
                elif priority_score > 6:
                    priority_class = 'priority-medium'
                else:
                    priority_class = 'priority-low'
                
                html += f"""
                <div class="action-item {priority_class}">
                    <h4>{i}. {action['action']}</h4>
                    <p><strong>Category:</strong> {category}</p>
                    <p><strong>Expected Impact:</strong> {action.get('expected_impact', 'TBD')}</p>
                    <p><strong>Priority Score:</strong> {priority_score:.1f}</p>
                    <p><strong>Resources:</strong> {', '.join(action.get('resources_needed', []))}</p>
                </div>
                """
        
        # Resource requirements
        resource_reqs = action_plan.get('resource_requirements', {})
        if resource_reqs:
            most_needed = resource_reqs.get('most_needed_resources', [])
            
            html += """
            <h3>Resource Requirements</h3>
            <div class="metric">
                <h4>🔧 Resource Allocation</h4>
            """
            
            html += f"<p><strong>Total Actions:</strong> {resource_reqs.get('total_actions', 0)}</p>"
            html += f"<p><strong>Immediate Priority:</strong> {resource_reqs.get('immediate_priority_actions', 0)}</p>"
            
            if most_needed:
                html += "<p><strong>Most Needed Resources:</strong></p><ul>"
                for resource, count in most_needed[:5]:
                    html += f"<li>{resource}: {count} actions</li>"
                html += "</ul>"
            
            html += "</div>"
        
        html += "</div>"
        return html

    def _create_risk_dashboard_chart(self, risk_assessment: Dict[str, Any], viz_dir: Path) -> str:
        """Create risk assessment dashboard"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Risk level gauge
            risk_level = risk_assessment.get('overall_risk_level', 'Unknown')
            risk_factors = risk_assessment.get('identified_risks', [])
            
            # Risk level chart
            levels = ['Low', 'Medium', 'High']
            risk_colors = ['#2ECC71', '#F39C12', '#E74C3C']
            current_index = levels.index(risk_level) if risk_level in levels else 1
            
            bars1 = ax1.bar(levels, [1, 1, 1], color=['lightgray' if i != current_index else risk_colors[i] for i in range(3)])
            bars1[current_index].set_edgecolor('black')
            bars1[current_index].set_linewidth(3)
            ax1.set_title(f'Overall Risk Level: {risk_level}', fontweight='bold')
            ax1.set_ylabel('Risk Assessment')
            
            # Risk factors breakdown
            if risk_factors:
                risk_types = [risk['risk_type'] for risk in risk_factors]
                risk_severities = [risk['severity'] for risk in risk_factors]
                
                severity_counts = {'High': 0, 'Medium': 0, 'Low': 0}
                for severity in risk_severities:
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                
                ax2.pie(severity_counts.values(), labels=severity_counts.keys(), autopct='%1.0f',
                    colors=['#E74C3C', '#F39C12', '#2ECC71'])
                ax2.set_title('Risk Factors by Severity', fontweight='bold')
            
            # Risk categories
            if risk_factors:
                risk_type_counts = {}
                for risk in risk_factors:
                    risk_type = risk['risk_type']
                    risk_type_counts[risk_type] = risk_type_counts.get(risk_type, 0) + 1
                
                if risk_type_counts:
                    ax3.barh(list(risk_type_counts.keys()), list(risk_type_counts.values()), color='coral')
                    ax3.set_title('Risk Categories', fontweight='bold')
                    ax3.set_xlabel('Number of Risk Factors')
            
            # Risk timeline/priority
            if risk_factors:
                priorities = [risk.get('severity', 'Medium') for risk in risk_factors]
                priority_timeline = {'Immediate': 0, 'Short-term': 0, 'Long-term': 0}
                
                for priority in priorities:
                    if priority == 'High':
                        priority_timeline['Immediate'] += 1
                    elif priority == 'Medium':
                        priority_timeline['Short-term'] += 1
                    else:
                        priority_timeline['Long-term'] += 1
                
                ax4.bar(priority_timeline.keys(), priority_timeline.values(), 
                    color=['#E74C3C', '#F39C12', '#2ECC71'])
                ax4.set_title('Risk Priority Timeline', fontweight='bold')
                ax4.set_ylabel('Number of Risks')
            
            plt.suptitle('Risk Assessment Dashboard', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = viz_dir / "risk_dashboard.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating risk dashboard chart: {e}")
            return ""

    def _create_market_dominance_chart(self, market_dominance: Dict[str, Any], viz_dir: Path) -> str:
        """Create market dominance bubble chart"""
        
        try:
            plt.figure(figsize=(12, 8))
            
            companies = []
            market_shares = []
            dominance_scores = []
            colors = ['#2E86C1', '#E74C3C', '#27AE60']
            
            for i, (company, data) in enumerate(market_dominance.items()):
                companies.append(company.title())
                market_shares.append(data.get('market_share', 0) * 100)  # Convert to percentage
                dominance_scores.append(data.get('dominance_score', 0) * 1000)  # Scale for bubble size
            
            # Create bubble chart
            scatter = plt.scatter(range(len(companies)), market_shares, 
                                s=dominance_scores, alpha=0.7, c=colors[:len(companies)])
            
            # Customize chart
            plt.xticks(range(len(companies)), companies)
            plt.ylabel('Market Share (%)', fontsize=12, fontweight='bold')
            plt.title('Market Dominance Analysis\n(Bubble size = Dominance Score)', fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels
            for i, (company, share, score) in enumerate(zip(companies, market_shares, dominance_scores)):
                plt.annotate(f'{share:.1f}%\nScore: {score/1000:.3f}', 
                        (i, share), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            chart_path = viz_dir / "market_dominance_bubble.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating market dominance chart: {e}")
            return ""
