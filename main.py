"""
Enhanced SEO Analysis - Complete Implementation with All Modular Updates
Incorporates: Advanced Analysis + Gap Analysis + Visualizations + Data Quality + Predictive Analytics + Strategic Planning
"""

from data_loader import SEODataLoader
from seo_analyzer import SEOAnalyzer  
from report_generator import ReportGenerator
from data_validator import DataValidator
import logging
from datetime import datetime

def setup_logging():
    """Setup logging for the analysis"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('seo_analysis.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Run complete enhanced SEO analysis with all advanced features"""
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 Advanced SEO Competitive Intelligence Analysis")
    print("📅 Data Period: May 19-21, 2025")
    print("🧱 Features: Advanced Analysis + Gap Analysis + Visualizations + Data Quality + Predictive Analytics + Strategic Planning")
    print("=" * 100)
    
    start_time = datetime.now()
    
    try:
        # =================================================================
        # STEP 1: DATA LOADING & VALIDATION (Brick 4)
        # =================================================================
        print("\n1️⃣ STEP 1: Data Loading & Quality Validation")
        print("-" * 60)
        
        # Load data
        loader = SEODataLoader()
        data = loader.load_all_data()
        
        # Validate data quality (Brick 4)
        validator = DataValidator()
        validation_results = validator.validate_all_data(data)
        
        # Check if data quality is sufficient
        overall_quality = validation_results['summary']['average_quality_score']
        print(f"📊 Overall Data Quality Score: {overall_quality:.2f}")
        
        if overall_quality < 0.5:
            print("❌ Data quality too low for reliable analysis!")
            return
        elif overall_quality < 0.7:
            print("⚠️ Data quality concerns detected - proceeding with caution")
        else:
            print("✅ Data quality is good - proceeding with full analysis")
        
        # =================================================================
        # STEP 2: CORE SEO ANALYSIS (Enhanced)
        # =================================================================
        print("\n2️⃣ STEP 2: Core SEO Analysis")
        print("-" * 60)
        
        analyzer = SEOAnalyzer()
        results = analyzer.analyze_all(data)
        
        # =================================================================
        # STEP 3: ADVANCED KEYWORD ANALYSIS (Brick 1)
        # =================================================================
        print("\n3️⃣ STEP 3: Advanced Keyword Analysis")
        print("-" * 60)
        
        advanced_results = analyzer.run_advanced_keyword_analysis(data)
        results['advanced_keyword_analysis'] = advanced_results
        
        # Print key insights from advanced analysis
        if 'intent_distribution' in advanced_results:
            intent_dist = advanced_results['intent_distribution']
            print(f"🎯 Intent Distribution:")
            for intent, count in intent_dist.items():
                print(f"   {intent.title()}: {count:,} keywords")
        
        if 'serp_features' in advanced_results:
            serp_coverage = advanced_results['serp_features']['coverage_percentage']
            print(f"🔍 SERP Features Coverage: {serp_coverage:.1f}%")
        
        # =================================================================
        # STEP 4: COMPETITIVE GAP ANALYSIS (Brick 2)
        # =================================================================
        print("\n4️⃣ STEP 4: Competitive Gap Analysis")
        print("-" * 60)
        
        gap_results = analyzer.run_detailed_gap_analysis(data)
        results['competitive_gaps'] = gap_results
        
        # Print gap analysis insights
        position_gaps = gap_results.get('position_gaps', [])
        print(f"📉 Position Gaps Identified: {len(position_gaps)}")
        
        quick_wins = gap_results.get('quick_wins', [])
        print(f"🎯 Quick Win Opportunities: {len(quick_wins)}")
        
        if quick_wins:
            print("   Top Quick Wins:")
            for i, win in enumerate(quick_wins[:3], 1):
                if win['type'] == 'page_2_keywords':
                    print(f"   {i}. {win['keyword']} (Position {win['current_position']}) - {win['traffic_potential']:.1f}% traffic")
        
        # =================================================================
        # STEP 5: ADVANCED COMPETITIVE METRICS & PREDICTIONS (Update 4)
        # =================================================================
        print("\n5️⃣ STEP 5: Advanced Competitive Metrics & Predictive Analytics")
        print("-" * 60)
        
        # Calculate advanced competitive metrics
        print("🔬 Calculating advanced competitive metrics...")
        advanced_metrics = analyzer.calculate_advanced_competitive_metrics(data)
        results['advanced_metrics'] = advanced_metrics
        
        # Print advanced insights
        threat_level = advanced_metrics.get('threat_assessment', {}).get('threat_level', 'Unknown')
        print(f"🚨 Competitive Threat Level: {threat_level}")
        
        hhi = advanced_metrics.get('lenovo_hhi', 0)
        concentration_risk = advanced_metrics.get('traffic_concentration_risk', 'Unknown')
        print(f"📊 Traffic Concentration Risk: {concentration_risk} (HHI: {hhi:.3f})")
        
        # Market dominance insights
        market_dominance = advanced_metrics.get('market_dominance', {})
        if 'lenovo' in market_dominance:
            lenovo_status = market_dominance['lenovo']['competitive_status']
            dominance_score = market_dominance['lenovo']['dominance_score']
            print(f"🏆 Competitive Status: {lenovo_status} (Score: {dominance_score:.3f})")
        
        # Generate predictive insights
        print("\n🔮 Generating Predictive Analytics...")
        predictions = analyzer.generate_predictive_insights(data, advanced_metrics)
        results['predictions'] = predictions
        
        # Print prediction highlights
        if 'traffic_forecast' in predictions:
            forecast_change = predictions['traffic_forecast'].get('forecast_change', 0)
            forecast_reliability = predictions['traffic_forecast'].get('forecast_reliability', 'Unknown')
            print(f"📈 30-Day Traffic Forecast: {forecast_change:+.1f}% change ({forecast_reliability} reliability)")
        
        if 'competitive_threats' in predictions:
            threat_count = len(predictions['competitive_threats'])
            print(f"⚠️ Active Competitive Threats: {threat_count}")
            
            for threat in predictions['competitive_threats'][:2]:  # Show top 2 threats
                print(f"   • {threat['competitor'].title()}: {threat['threat_level']} threat level")
        
        # Market share trajectory
        if 'market_share_trajectory' in predictions:
            trajectory = predictions['market_share_trajectory']
            current_share = trajectory.get('current_share', 0)
            projected_change = trajectory.get('projected_3_month_change', 0)
            print(f"📊 Market Share Trajectory: {current_share:.1f}% current → {projected_change:+.1f}% projected change")
        
        # Risk assessment summary
        if 'risk_assessment' in predictions:
            risk_level = predictions['risk_assessment'].get('overall_risk_level', 'Unknown')
            risk_count = len(predictions['risk_assessment'].get('identified_risks', []))
            print(f"🛡️ Overall Risk Level: {risk_level} ({risk_count} factors identified)")
        
        # Keyword opportunities insight
        if 'keyword_opportunities' in predictions:
            keyword_opps = predictions['keyword_opportunities']
            high_prob_wins = keyword_opps.get('high_probability_wins', 0)
            expected_traffic = keyword_opps.get('expected_traffic_gain', 0)
            print(f"🎯 Keyword Opportunities: {high_prob_wins} high-probability wins, {expected_traffic:,.0f} expected traffic gain")
        
        # =================================================================
        # STEP 6: STRATEGIC ACTION PLAN GENERATION (Update 6)
        # =================================================================
        print("\n6️⃣ STEP 6: Strategic Action Plan Generation")
        print("-" * 60)
        
        # Generate comprehensive strategic action plan
        print("📋 Generating strategic action plan...")
        action_plan = analyzer.generate_strategic_action_plan(results, advanced_metrics, predictions)
        results['action_plan'] = action_plan
        
        # Print action plan highlights
        immediate_actions = action_plan.get('immediate_actions', [])
        defensive_actions = action_plan.get('defensive_actions', [])
        offensive_actions = action_plan.get('offensive_actions', [])
        prioritized_actions = action_plan.get('prioritized_actions', [])
        
        print(f"⚡ Immediate Actions: {len(immediate_actions)}")
        print(f"🛡️ Defensive Actions: {len(defensive_actions)}")
        print(f"🚀 Offensive Actions: {len(offensive_actions)}")
        
        if prioritized_actions:
            print("\n🎯 Top Priority Actions:")
            for i, action in enumerate(prioritized_actions[:3], 1):
                print(f"   {i}. {action['action']} (Priority Score: {action.get('priority_score', 0):.1f})")
                print(f"      Category: {action.get('category', 'unknown').replace('_', ' ').title()}")
                print(f"      Impact: {action.get('expected_impact', 'TBD')}")
        
        # Resource requirements summary
        resource_reqs = action_plan.get('resource_requirements', {})
        if resource_reqs:
            most_needed = resource_reqs.get('most_needed_resources', [])
            print(f"\n🔧 Resource Requirements:")
            print(f"   Total Actions: {resource_reqs.get('total_actions', 0)}")
            print(f"   Immediate Priority: {resource_reqs.get('immediate_priority_actions', 0)}")
            if most_needed:
                print(f"   Most Needed Resources: {', '.join([r[0] for r in most_needed[:3]])}")
        
        # =================================================================
        # STEP 7: ENHANCED REPORT GENERATION WITH ADVANCED VISUALIZATIONS
        # =================================================================
        print("\n7️⃣ STEP 7: Enhanced Report Generation & Advanced Visualizations")
        print("-" * 60)
        
        # Add validation results to the final results
        results['data_validation'] = validation_results
        
        # Generate comprehensive reports
        reporter = ReportGenerator()
        reports = reporter.generate_all_reports(results)
        
        # Generate standard visualizations
        print("📊 Generating standard visualizations...")
        visualizations = reporter.generate_visualizations(results)
        
        # Generate advanced visualizations with predictions (Update 3)
        print("🔬 Generating advanced predictive visualizations...")
        try:
            advanced_visualizations = reporter.create_advanced_visualizations(results, advanced_metrics, predictions)
            # Combine all visualizations
            all_visualizations = {**visualizations, **advanced_visualizations}
        except Exception as e:
            print(f"⚠️ Advanced visualizations not available: {e}")
            all_visualizations = visualizations
        
        reports['visualizations'] = all_visualizations
        
        print(f"📊 Generated {len(all_visualizations)} total visualizations")
        
        # =================================================================
        # STEP 8: EXECUTIVE SUMMARY & STRATEGIC BRIEFING
        # =================================================================
        print("\n8️⃣ STEP 8: Executive Summary & Strategic Briefing")
        print("-" * 60)
        
        executive_summary = generate_enhanced_executive_summary(results, validation_results, advanced_metrics, predictions, action_plan)
        print_enhanced_executive_summary(executive_summary)
        
        # =================================================================
        # COMPLETION SUMMARY
        # =================================================================
        execution_time = datetime.now() - start_time
        
        print(f"\n🎉 Advanced SEO Analysis Complete!")
        print("=" * 70)
        print(f"⏱️ Total Execution Time: {execution_time.total_seconds():.1f} seconds")
        
        # Analysis completeness metrics
        analysis_completeness = calculate_analysis_completeness(results)
        print(f"📈 Analysis Completeness: {analysis_completeness:.1%}")
        
        print(f"\n📊 Generated Outputs:")
        for report_type, path in reports.items():
            if path:  # Only show successful reports
                icon = "📄" if report_type != "visualizations" else "📊"
                print(f"   {icon} {report_type.upper()}: {path}")
        
        # Enhanced metrics summary
        print(f"\n📈 Enhanced Metrics Summary:")
        summary = results.get('summary', {})
        if 'lenovo' in summary:
            lenovo = summary['lenovo']
            print(f"   📋 Keywords Analyzed: {lenovo.get('total_keywords', 0):,}")
            print(f"   📍 Average Position: {lenovo.get('avg_position', 0):.1f}")
            print(f"   🚀 Traffic Share: {lenovo.get('total_traffic', 0):.1f}%")
            print(f"   🏆 Top 10 Rankings: {lenovo.get('top_10_count', 0):,}")
            
            # Market metrics
            if 'competitive' in results:
                market_share = results['competitive'].get('market_share', {}).get('lenovo', 0)
                print(f"   📊 Market Share: {market_share:.1f}%")
            
            # Advanced metrics
            if 'advanced_keyword_analysis' in results:
                branded_analysis = results['advanced_keyword_analysis'].get('branded_vs_nonbranded', {})
                if branded_analysis:
                    branded_count = branded_analysis.get('branded_count', 0)
                    non_branded_count = branded_analysis.get('non_branded_count', 0)
                    print(f"   🏷️ Branded Keywords: {branded_count:,}")
                    print(f"   🔍 Non-Branded Keywords: {non_branded_count:,}")
            
            # Opportunities and threats
            if 'competitive_gaps' in results:
                quick_wins_count = len(results['competitive_gaps'].get('quick_wins', []))
                print(f"   ⚡ Quick Win Opportunities: {quick_wins_count}")
            
            if 'predictions' in results:
                threat_count = len(results['predictions'].get('competitive_threats', []))
                print(f"   ⚠️ Active Threats: {threat_count}")
            
            # Strategic actions
            if 'action_plan' in results:
                total_actions = results['action_plan'].get('resource_requirements', {}).get('total_actions', 0)
                immediate_actions = results['action_plan'].get('resource_requirements', {}).get('immediate_priority_actions', 0)
                print(f"   📋 Strategic Actions: {total_actions} total, {immediate_actions} immediate")
        
        # Data quality summary
        print(f"\n📊 Data Quality & Analysis Summary:")
        print(f"   ✅ Overall Quality Score: {overall_quality:.2f}/1.00")
        print(f"   📁 Datasets Processed: {validation_results['summary']['total_datasets']}")
        print(f"   📄 Total Rows Analyzed: {validation_results['summary']['total_rows']:,}")
        print(f"   🔬 Advanced Metrics Calculated: {len(advanced_metrics)} categories")
        print(f"   🔮 Predictions Generated: {len(predictions)} categories")
        
        # Final recommendations highlight
        if 'action_plan' in results and 'prioritized_actions' in results['action_plan']:
            priority_action = results['action_plan']['prioritized_actions'][0] if results['action_plan']['prioritized_actions'] else None
            if priority_action:
                print(f"\n🎯 Next Priority Action: {priority_action['action']}")
                print(f"   Expected Impact: {priority_action.get('expected_impact', 'TBD')}")
        
        logger.info("Advanced SEO analysis with predictive analytics completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis Error: {e}")
        import traceback
        traceback.print_exc()

def generate_enhanced_executive_summary(results, validation_results, advanced_metrics, predictions, action_plan):
    """Generate enhanced executive summary with all advanced insights"""
    
    summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_quality_score': validation_results['summary']['average_quality_score'],
        'datasets_analyzed': validation_results['summary']['total_datasets'],
        'total_keywords_analyzed': validation_results['summary']['total_rows']
    }
    
    # Core performance metrics
    if 'summary' in results and 'lenovo' in results['summary']:
        lenovo_data = results['summary']['lenovo']
        summary.update({
            'lenovo_keywords': lenovo_data.get('total_keywords', 0),
            'average_position': lenovo_data.get('avg_position', 0),
            'traffic_share': lenovo_data.get('total_traffic', 0),
            'top_10_rankings': lenovo_data.get('top_10_count', 0)
        })
    
    # Market share analysis
    if 'competitive' in results:
        market_share = results['competitive'].get('market_share', {})
        summary['market_share'] = market_share.get('lenovo', 0)
    
    # Advanced insights
    if 'advanced_keyword_analysis' in results:
        advanced = results['advanced_keyword_analysis']
        summary['serp_coverage'] = advanced.get('serp_features', {}).get('coverage_percentage', 0)
        
        intent_dist = advanced.get('intent_distribution', {})
        summary['commercial_keywords'] = intent_dist.get('commercial', 0)
        summary['branded_keywords'] = intent_dist.get('branded', 0)
    
    # Gap analysis insights
    if 'competitive_gaps' in results:
        gaps = results['competitive_gaps']
        summary['position_gaps'] = len(gaps.get('position_gaps', []))
        summary['quick_wins'] = len(gaps.get('quick_wins', []))
    
    # Advanced competitive metrics
    summary.update({
        'threat_level': advanced_metrics.get('threat_assessment', {}).get('threat_level', 'Unknown'),
        'competitive_status': advanced_metrics.get('market_dominance', {}).get('lenovo', {}).get('competitive_status', 'Unknown'),
        'traffic_concentration_risk': advanced_metrics.get('traffic_concentration_risk', 'Unknown')
    })
    
    # Predictions
    if 'traffic_forecast' in predictions:
        summary['traffic_forecast_change'] = predictions['traffic_forecast'].get('forecast_change', 0)
        summary['forecast_reliability'] = predictions['traffic_forecast'].get('forecast_reliability', 'Unknown')
    
    if 'competitive_threats' in predictions:
        summary['active_threats'] = len(predictions['competitive_threats'])
    
    if 'risk_assessment' in predictions:
        summary['overall_risk_level'] = predictions['risk_assessment'].get('overall_risk_level', 'Unknown')
        summary['risk_factors_count'] = len(predictions['risk_assessment'].get('identified_risks', []))
    
    # Strategic actions
    if action_plan:
        summary['total_strategic_actions'] = action_plan.get('resource_requirements', {}).get('total_actions', 0)
        summary['immediate_actions'] = action_plan.get('resource_requirements', {}).get('immediate_priority_actions', 0)
        summary['defensive_actions'] = action_plan.get('resource_requirements', {}).get('defensive_priority_actions', 0)
    
    return summary

def print_enhanced_executive_summary(summary):
    """Print enhanced formatted executive summary"""
    
    print("📋 ADVANCED EXECUTIVE SUMMARY")
    print("=" * 50)
    print(f"Analysis Date: {summary['analysis_date']}")
    print(f"Data Quality: {summary['data_quality_score']:.2f}/1.00")
    print(f"Datasets: {summary['datasets_analyzed']}")
    print(f"Total Keywords: {summary.get('total_keywords_analyzed', 0):,}")
    
    print("\n🏢 LENOVO PERFORMANCE")
    print("-" * 30)
    print(f"Keywords Analyzed: {summary.get('lenovo_keywords', 0):,}")
    print(f"Average Position: {summary.get('average_position', 0):.1f}")
    print(f"Traffic Share: {summary.get('traffic_share', 0):.1f}%")
    print(f"Top 10 Rankings: {summary.get('top_10_rankings', 0):,}")
    print(f"Market Share: {summary.get('market_share', 0):.1f}%")
    
    print("\n🎯 ADVANCED INSIGHTS")
    print("-" * 25)
    print(f"SERP Coverage: {summary.get('serp_coverage', 0):.1f}%")
    print(f"Commercial Keywords: {summary.get('commercial_keywords', 0):,}")
    print(f"Branded Keywords: {summary.get('branded_keywords', 0):,}")
    print(f"Competitive Status: {summary.get('competitive_status', 'Unknown')}")
    print(f"Traffic Risk Level: {summary.get('traffic_concentration_risk', 'Unknown')}")
    
    print("\n🔮 PREDICTIVE ANALYTICS")
    print("-" * 28)
    print(f"30-Day Traffic Forecast: {summary.get('traffic_forecast_change', 0):+.1f}%")
    print(f"Forecast Reliability: {summary.get('forecast_reliability', 'Unknown')}")
    print(f"Threat Level: {summary.get('threat_level', 'Unknown')}")
    print(f"Active Threats: {summary.get('active_threats', 0)}")
    print(f"Risk Level: {summary.get('overall_risk_level', 'Unknown')} ({summary.get('risk_factors_count', 0)} factors)")
    
    print("\n⚡ OPPORTUNITIES & ACTIONS")
    print("-" * 30)
    print(f"Position Gaps: {summary.get('position_gaps', 0)}")
    print(f"Quick Wins: {summary.get('quick_wins', 0)}")
    print(f"Strategic Actions: {summary.get('total_strategic_actions', 0)} total")
    print(f"Immediate Actions: {summary.get('immediate_actions', 0)}")
    print(f"Defensive Actions: {summary.get('defensive_actions', 0)}")

def calculate_analysis_completeness(results):
    """Calculate how complete the analysis is"""
    
    expected_components = [
        'summary', 'competitive', 'keyword_analysis', 'opportunity_analysis',
        'advanced_keyword_analysis', 'competitive_gaps', 'advanced_metrics',
        'predictions', 'action_plan', 'data_validation'
    ]
    
    completed_components = len([comp for comp in expected_components if comp in results and results[comp]])
    return completed_components / len(expected_components)

if __name__ == "__main__":
    main()
