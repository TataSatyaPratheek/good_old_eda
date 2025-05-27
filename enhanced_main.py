"""
Enhanced Main Module with TOFU Analysis Integration
Complete implementation with all advanced modules
"""

from data_loader import SEODataLoader
from seo_analyzer import SEOAnalyzer  
from report_generator import ReportGenerator
from data_validator import DataValidator
from tofu_analyzer import TOFUAnalyzer
from brand_analyzer import BrandAnalyzer
from traffic_diagnostics import TrafficDiagnostics
from competitor_intelligence import CompetitorIntelligence
from correlation_engine import CorrelationEngine
from alert_system import AlertSystem
import logging
from datetime import datetime

def setup_logging():
    """Setup comprehensive logging for the analysis"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('seo_analysis.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Run complete enhanced SEO analysis with all TOFU and advanced features"""
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 Complete SEO Intelligence Platform with TOFU Analysis")
    print("📅 Data Period: May 19-21, 2025")
    print("🧱 Features: TOFU Analysis + Brand Analysis + Traffic Diagnostics + Competitive Intelligence + Correlation Analysis + Real-time Alerts")
    print("=" * 120)
    
    start_time = datetime.now()
    
    try:
        # =================================================================
        # STEP 1: DATA LOADING & VALIDATION
        # =================================================================
        print("\n1️⃣ STEP 1: Data Loading & Quality Validation")
        print("-" * 60)
        
        loader = SEODataLoader()
        data = loader.load_all_data()
        
        validator = DataValidator()
        validation_results = validator.validate_all_data(data)
        
        overall_quality = validation_results['summary']['average_quality_score']
        print(f"📊 Overall Data Quality Score: {overall_quality:.2f}")
        
        if overall_quality < 0.5:
            print("❌ Data quality too low for reliable analysis!")
            return
        
        # =================================================================
        # STEP 2: CORE SEO ANALYSIS
        # =================================================================
        print("\n2️⃣ STEP 2: Core SEO Analysis")
        print("-" * 60)
        
        analyzer = SEOAnalyzer()
        results = analyzer.analyze_all(data)
        
        # =================================================================
        # STEP 3: TOFU ANALYSIS (NEW)
        # =================================================================
        print("\n3️⃣ STEP 3: TOFU (Top of Funnel) Analysis")
        print("-" * 60)
        
        tofu_analyzer = TOFUAnalyzer()
        tofu_results = tofu_analyzer.analyze_tofu_performance(data)
        results['tofu_analysis'] = tofu_results
        
        # Print TOFU insights
        tofu_score = tofu_results.get('tofu_score', {}).get('overall_score', 0)
        performance_grade = tofu_results.get('tofu_score', {}).get('performance_grade', 'Unknown')
        print(f"🎯 TOFU Performance Score: {tofu_score:.1f}/100 (Grade: {performance_grade})")
        
        non_branded_data = tofu_results.get('non_branded_analysis', {}).get('lenovo', {})
        if non_branded_data:
            acquisition_potential = non_branded_data.get('customer_acquisition_potential', 0)
            print(f"🔍 Customer Acquisition Potential: {acquisition_potential:.1f}/100")
            
            top_acquisition = non_branded_data.get('top_acquisition_keywords', [])
            if top_acquisition:
                print(f"⚡ Top Acquisition Opportunities: {len(top_acquisition)} keywords identified")
        
        # =================================================================
        # STEP 4: BRAND ANALYSIS (NEW)
        # =================================================================
        print("\n4️⃣ STEP 4: Brand Analysis & Awareness")
        print("-" * 60)
        
        brand_analyzer = BrandAnalyzer()
        brand_results = brand_analyzer.analyze_brand_performance(data)
        results['brand_analysis'] = brand_results
        
        # Print brand insights
        brand_health = brand_results.get('brand_health_score', {})
        if brand_health:
            health_score = brand_health.get('overall_health_score', 0)
            health_grade = brand_health.get('health_grade', 'Unknown')
            print(f"🏷️ Brand Health Score: {health_score:.1f}/100 (Grade: {health_grade})")
        
        brand_awareness = brand_results.get('brand_awareness_metrics', {}).get('lenovo', {})
        if brand_awareness:
            awareness_score = brand_awareness.get('brand_awareness_score', 0)
            share_of_voice = brand_awareness.get('share_of_voice', 0)
            print(f"📢 Brand Awareness Score: {awareness_score:.1f}/100")
            print(f"🎤 Share of Voice: {share_of_voice:.1f}%")
        
        # =================================================================
        # STEP 5: TRAFFIC DIAGNOSTICS (NEW)
        # =================================================================
        print("\n5️⃣ STEP 5: Traffic Diagnostics & Pattern Analysis")
        print("-" * 60)
        
        traffic_diagnostics = TrafficDiagnostics()
        # For demo, we'll use current data as both current and historical
        diagnostics_results = traffic_diagnostics.diagnose_traffic_patterns(data, data)
        results['traffic_diagnostics'] = diagnostics_results
        
        # Print diagnostics insights
        recent_changes = diagnostics_results.get('recent_changes', {})
        significant_drops = recent_changes.get('significant_drops', [])
        significant_gains = recent_changes.get('significant_gains', [])
        print(f"📉 Traffic Drops Detected: {len(significant_drops)}")
        print(f"📈 Traffic Gains Detected: {len(significant_gains)}")
        
        monitoring_status = diagnostics_results.get('monitoring_status', 'unknown')
        print(f"🔍 Overall Monitoring Status: {monitoring_status}")
        
        # =================================================================
        # STEP 6: COMPETITOR INTELLIGENCE (NEW)
        # =================================================================
        print("\n6️⃣ STEP 6: Advanced Competitor Intelligence")
        print("-" * 60)
        
        competitor_intel = CompetitorIntelligence()
        intel_results = competitor_intel.analyze_competitor_strategies(data)
        results['competitor_intelligence'] = intel_results
        
        # Print competitor insights
        competitor_overview = intel_results.get('competitor_overview', {})
        competitive_intensity = competitor_overview.get('competitive_intensity', {})
        if competitive_intensity:
            intensity_level = competitive_intensity.get('overall_intensity', 'unknown')
            print(f"⚔️ Competitive Intensity: {intensity_level}")
        
        strategy_insights = intel_results.get('strategy_insights', [])
        threat_insights = [insight for insight in strategy_insights if insight.get('insight_type') == 'strength']
        opportunity_insights = [insight for insight in strategy_insights if insight.get('insight_type') == 'weakness']
        print(f"🚨 Competitive Threats Identified: {len(threat_insights)}")
        print(f"💡 Competitive Opportunities: {len(opportunity_insights)}")
        
        # =================================================================
        # STEP 7: CORRELATION ANALYSIS (NEW)
        # =================================================================
        print("\n7️⃣ STEP 7: Advanced Correlation Analysis")
        print("-" * 60)
        
        correlation_engine = CorrelationEngine()
        correlation_results = correlation_engine.analyze_metric_correlations(data)
        results['correlation_analysis'] = correlation_results
        
        # Print correlation insights
        correlation_insights = correlation_results.get('correlation_insights', [])
        primary_correlations = correlation_results.get('primary_correlations', {})
        
        print(f"🔗 Correlation Insights Generated: {len(correlation_insights)}")
        
        # Show position-traffic correlation for Lenovo
        lenovo_pos_corr = primary_correlations.get('position_traffic_correlation', {}).get('lenovo', {})
        if lenovo_pos_corr:
            correlation = lenovo_pos_corr.get('correlation', 0)
            significance = lenovo_pos_corr.get('significance', 'unknown')
            print(f"📊 Position-Traffic Correlation: {correlation:.3f} ({significance})")
        
        # =================================================================
        # STEP 8: REAL-TIME ALERT SYSTEM (NEW)
        # =================================================================
        print("\n8️⃣ STEP 8: Real-time Alert System")
        print("-" * 60)
        
        alert_system = AlertSystem()
        alert_results = alert_system.monitor_and_alert(data)
        results['alert_system'] = alert_results
        
        # Print alert insights
        new_alerts = alert_results.get('new_alerts', [])
        active_alerts = alert_results.get('active_alerts', [])
        alert_summary = alert_results.get('alert_summary', {})
        
        print(f"🚨 New Alerts Generated: {len(new_alerts)}")
        print(f"⚠️ Total Active Alerts: {len(active_alerts)}")
        
        # Show alert breakdown
        alerts_by_severity = alert_summary.get('alerts_by_severity', {})
        for severity, count in alerts_by_severity.items():
            if count > 0:
                icon = "🔴" if severity == "critical" else "🟠" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                print(f"   {icon} {severity.title()}: {count}")
        
        # =================================================================
        # STEP 9: ENHANCED REPORT GENERATION
        # =================================================================
        print("\n9️⃣ STEP 9: Enhanced Report Generation")
        print("-" * 60)
        
        results['data_validation'] = validation_results
        
        reporter = ReportGenerator()
        reports = reporter.generate_all_reports(results)
        
        # Generate comprehensive visualizations
        print("📊 Generating comprehensive visualizations...")
        visualizations = reporter.generate_visualizations(results)
        
        # Generate advanced visualizations
        print("🔬 Generating advanced visualizations...")
        try:
            advanced_visualizations = reporter.create_advanced_visualizations(
                results, 
                results.get('advanced_metrics', {}), 
                results.get('predictions', {})
            )
            all_visualizations = {**visualizations, **advanced_visualizations}
        except Exception as e:
            print(f"⚠️ Some advanced visualizations not available: {e}")
            all_visualizations = visualizations
        
        reports['visualizations'] = all_visualizations
        print(f"📊 Generated {len(all_visualizations)} total visualizations")
        
        # =================================================================
        # STEP 10: COMPREHENSIVE EXECUTIVE SUMMARY
        # =================================================================
        print("\n🔟 STEP 10: Comprehensive Executive Summary")
        print("-" * 60)
        
        executive_summary = generate_comprehensive_executive_summary(results)
        print_comprehensive_executive_summary(executive_summary)
        
        # =================================================================
        # COMPLETION SUMMARY
        # =================================================================
        execution_time = datetime.now() - start_time
        
        print(f"\n🎉 Complete SEO Intelligence Analysis Finished!")
        print("=" * 80)
        print(f"⏱️ Total Execution Time: {execution_time.total_seconds():.1f} seconds")
        
        # Comprehensive metrics summary
        print(f"\n📈 Comprehensive Analysis Summary:")
        summary = results.get('summary', {})
        if 'lenovo' in summary:
            lenovo = summary['lenovo']
            print(f"   📋 Keywords Analyzed: {lenovo.get('total_keywords', 0):,}")
            print(f"   📍 Average Position: {lenovo.get('avg_position', 0):.1f}")
            print(f"   🚀 Traffic Share: {lenovo.get('total_traffic', 0):.1f}%")
            print(f"   🏆 Top 10 Rankings: {lenovo.get('top_10_count', 0):,}")
        
        # TOFU metrics
        if tofu_score:
            print(f"   🎯 TOFU Score: {tofu_score:.1f}/100")
            
        # Brand metrics
        if brand_health:
            print(f"   🏷️ Brand Health: {brand_health.get('overall_health_score', 0):.1f}/100")
        
        # Alert metrics
        print(f"   🚨 Active Alerts: {len(active_alerts)}")
        print(f"   📊 Correlations Analyzed: {len(correlation_insights)}")
        
        # Analysis completeness
        analysis_completeness = calculate_analysis_completeness(results)
        print(f"\n📊 Analysis Completeness: {analysis_completeness:.1%}")
        
        print(f"\n📊 Generated Outputs:")
        for report_type, path in reports.items():
            if path:
                icon = "📄" if report_type != "visualizations" else "📊"
                print(f"   {icon} {report_type.upper()}: {path}")
        
        # Final strategic recommendation
        print(f"\n🎯 Next Strategic Focus:")
        
        # Determine top priority based on all analyses
        if tofu_score < 60:
            print("   Priority 1: Improve TOFU performance for better customer acquisition")
        elif len([a for a in new_alerts if a.severity.value in ['critical', 'high']]) > 0:
            print("   Priority 1: Address critical/high severity alerts immediately")
        elif brand_health.get('overall_health_score', 0) < 70:
            print("   Priority 1: Strengthen brand presence and awareness")
        else:
            print("   Priority 1: Maintain current performance and expand market share")
        
        logger.info("Complete SEO intelligence analysis with TOFU integration completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis Error: {e}")
        import traceback
        traceback.print_exc()

def generate_comprehensive_executive_summary(results):
    """Generate comprehensive executive summary with all modules"""
    
    summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_quality_score': results.get('data_validation', {}).get('summary', {}).get('average_quality_score', 0),
        'datasets_analyzed': results.get('data_validation', {}).get('summary', {}).get('total_datasets', 0),
        'total_keywords_analyzed': results.get('data_validation', {}).get('summary', {}).get('total_rows', 0)
    }
    
    # Core performance
    if 'summary' in results and 'lenovo' in results['summary']:
        lenovo_data = results['summary']['lenovo']
        summary.update({
            'lenovo_keywords': lenovo_data.get('total_keywords', 0),
            'average_position': lenovo_data.get('avg_position', 0),
            'traffic_share': lenovo_data.get('total_traffic', 0),
            'top_10_rankings': lenovo_data.get('top_10_count', 0)
        })
    
    # TOFU analysis
    tofu_data = results.get('tofu_analysis', {})
    if tofu_data:
        tofu_score = tofu_data.get('tofu_score', {}).get('overall_score', 0)
        summary['tofu_score'] = tofu_score
        summary['tofu_grade'] = tofu_data.get('tofu_score', {}).get('performance_grade', 'Unknown')
        
        non_branded = tofu_data.get('non_branded_analysis', {}).get('lenovo', {})
        if non_branded:
            summary['acquisition_potential'] = non_branded.get('customer_acquisition_potential', 0)
    
    # Brand analysis
    brand_data = results.get('brand_analysis', {})
    if brand_data:
        brand_health = brand_data.get('brand_health_score', {})
        summary['brand_health_score'] = brand_health.get('overall_health_score', 0)
        summary['brand_health_grade'] = brand_health.get('health_grade', 'Unknown')
        
        brand_awareness = brand_data.get('brand_awareness_metrics', {}).get('lenovo', {})
        if brand_awareness:
            summary['brand_awareness_score'] = brand_awareness.get('brand_awareness_score', 0)
            summary['share_of_voice'] = brand_awareness.get('share_of_voice', 0)
    
    # Traffic diagnostics
    traffic_data = results.get('traffic_diagnostics', {})
    if traffic_data:
        summary['monitoring_status'] = traffic_data.get('monitoring_status', 'unknown')
        recent_changes = traffic_data.get('recent_changes', {})
        summary['traffic_drops'] = len(recent_changes.get('significant_drops', []))
        summary['traffic_gains'] = len(recent_changes.get('significant_gains', []))
    
    # Competitor intelligence
    intel_data = results.get('competitor_intelligence', {})
    if intel_data:
        strategy_insights = intel_data.get('strategy_insights', [])
        summary['competitive_threats'] = len([i for i in strategy_insights if i.get('insight_type') == 'strength'])
        summary['competitive_opportunities'] = len([i for i in strategy_insights if i.get('insight_type') == 'weakness'])
    
    # Alert system
    alert_data = results.get('alert_system', {})
    if alert_data:
        summary['active_alerts'] = len(alert_data.get('active_alerts', []))
        summary['new_alerts'] = len(alert_data.get('new_alerts', []))
        
        alert_summary = alert_data.get('alert_summary', {})
        severity_counts = alert_summary.get('alerts_by_severity', {})
        summary['critical_alerts'] = severity_counts.get('critical', 0)
        summary['high_alerts'] = severity_counts.get('high', 0)
    
    # Correlation analysis
    correlation_data = results.get('correlation_analysis', {})
    if correlation_data:
        summary['correlation_insights'] = len(correlation_data.get('correlation_insights', []))
    
    return summary

def print_comprehensive_executive_summary(summary):
    """Print comprehensive formatted executive summary"""
    
    print("📋 COMPREHENSIVE EXECUTIVE SUMMARY")
    print("=" * 60)
    print(f"Analysis Date: {summary['analysis_date']}")
    print(f"Data Quality: {summary['data_quality_score']:.2f}/1.00")
    print(f"Datasets: {summary['datasets_analyzed']}")
    print(f"Total Keywords: {summary.get('total_keywords_analyzed', 0):,}")
    
    print("\n🏢 LENOVO PERFORMANCE")
    print("-" * 35)
    print(f"Keywords Analyzed: {summary.get('lenovo_keywords', 0):,}")
    print(f"Average Position: {summary.get('average_position', 0):.1f}")
    print(f"Traffic Share: {summary.get('traffic_share', 0):.1f}%")
    print(f"Top 10 Rankings: {summary.get('top_10_rankings', 0):,}")
    
    print("\n🎯 TOFU (TOP OF FUNNEL) ANALYSIS")
    print("-" * 40)
    print(f"TOFU Score: {summary.get('tofu_score', 0):.1f}/100 ({summary.get('tofu_grade', 'Unknown')})")
    print(f"Acquisition Potential: {summary.get('acquisition_potential', 0):.1f}/100")
    
    print("\n🏷️ BRAND ANALYSIS")
    print("-" * 25)
    print(f"Brand Health: {summary.get('brand_health_score', 0):.1f}/100 ({summary.get('brand_health_grade', 'Unknown')})")
    print(f"Brand Awareness: {summary.get('brand_awareness_score', 0):.1f}/100")
    print(f"Share of Voice: {summary.get('share_of_voice', 0):.1f}%")
    
    print("\n📊 TRAFFIC & MONITORING")
    print("-" * 30)
    print(f"Monitoring Status: {summary.get('monitoring_status', 'Unknown').title()}")
    print(f"Traffic Drops: {summary.get('traffic_drops', 0)}")
    print(f"Traffic Gains: {summary.get('traffic_gains', 0)}")
    
    print("\n⚔️ COMPETITIVE INTELLIGENCE")
    print("-" * 35)
    print(f"Competitive Threats: {summary.get('competitive_threats', 0)}")
    print(f"Competitive Opportunities: {summary.get('competitive_opportunities', 0)}")
    
    print("\n🚨 ALERT SYSTEM")
    print("-" * 20)
    print(f"Active Alerts: {summary.get('active_alerts', 0)}")
    print(f"New Alerts: {summary.get('new_alerts', 0)}")
    print(f"Critical Alerts: {summary.get('critical_alerts', 0)}")
    print(f"High Priority Alerts: {summary.get('high_alerts', 0)}")
    
    print("\n🔗 ANALYSIS INSIGHTS")
    print("-" * 25)
    print(f"Correlation Insights: {summary.get('correlation_insights', 0)}")

def calculate_analysis_completeness(results):
    """Calculate comprehensive analysis completeness"""
    
    expected_components = [
        'summary', 'competitive', 'keyword_analysis', 'opportunity_analysis',
        'tofu_analysis', 'brand_analysis', 'traffic_diagnostics',
        'competitor_intelligence', 'correlation_analysis', 'alert_system',
        'data_validation'
    ]
    
    completed_components = len([comp for comp in expected_components if comp in results and results[comp]])
    return completed_components / len(expected_components)

if __name__ == "__main__":
    main()
