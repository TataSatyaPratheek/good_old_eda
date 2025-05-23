"""
Enhanced SEO Analysis - Brick 1-4 Implementation
Incorporates: Advanced Analysis, Gap Analysis, Visualizations, Data Quality
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
    """Run enhanced SEO analysis with all Brick 1-4 features"""
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 Enhanced SEO Competitive Intelligence Analysis")
    print("📅 Data Period: May 19-21, 2025")
    print("🧱 Features: Advanced Analysis + Gap Analysis + Visualizations + Data Quality")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # =================================================================
        # STEP 1: DATA LOADING & VALIDATION (Brick 4)
        # =================================================================
        print("\n1️⃣ STEP 1: Data Loading & Quality Validation")
        print("-" * 50)
        
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
        print("-" * 50)
        
        analyzer = SEOAnalyzer()
        results = analyzer.analyze_all(data)
        
        # =================================================================
        # STEP 3: ADVANCED KEYWORD ANALYSIS (Brick 1)
        # =================================================================
        print("\n3️⃣ STEP 3: Advanced Keyword Analysis")
        print("-" * 50)
        
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
        print("-" * 50)
        
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
        # STEP 5: REPORT GENERATION WITH VISUALIZATIONS (Brick 3)
        # =================================================================
        print("\n5️⃣ STEP 5: Enhanced Report Generation")
        print("-" * 50)
        
        # Add validation results to the final results
        results['data_validation'] = validation_results
        
        # Generate comprehensive reports
        reporter = ReportGenerator()
        reports = reporter.generate_all_reports(results)
        
        # Generate visualizations (Brick 3)
        print("📊 Generating visualizations...")
        visualizations = reporter.generate_visualizations(results)
        reports['visualizations'] = visualizations
        
        # =================================================================
        # STEP 6: EXECUTIVE SUMMARY
        # =================================================================
        print("\n6️⃣ STEP 6: Executive Summary")
        print("-" * 50)
        
        executive_summary = generate_executive_summary(results, validation_results)
        print_executive_summary(executive_summary)
        
        # =================================================================
        # COMPLETION SUMMARY
        # =================================================================
        execution_time = datetime.now() - start_time
        
        print(f"\n🎉 Enhanced Analysis Complete!")
        print("=" * 50)
        print(f"⏱️ Execution Time: {execution_time.total_seconds():.1f} seconds")
        print(f"📊 Reports & Visualizations Generated:")
        
        for report_type, path in reports.items():
            if path:  # Only show successful reports
                print(f"   📄 {report_type.upper()}: {path}")
        
        # Quick stats with enhanced metrics
        summary = results.get('summary', {})
        if 'lenovo' in summary:
            lenovo = summary['lenovo']
            print(f"\n📈 Lenovo Enhanced Quick Stats:")
            print(f"   📋 Keywords: {lenovo.get('total_keywords', 0):,}")
            print(f"   📍 Avg Position: {lenovo.get('avg_position', 0):.1f}")
            print(f"   🚀 Traffic Share: {lenovo.get('total_traffic', 0):.1f}%")
            print(f"   🏆 Top 10 Rankings: {lenovo.get('top_10_count', 0):,}")
            
            # Additional advanced metrics
            if 'advanced_keyword_analysis' in results:
                branded_analysis = results['advanced_keyword_analysis'].get('branded_vs_nonbranded', {})
                if branded_analysis:
                    print(f"   🏷️ Branded Keywords: {branded_analysis.get('branded_count', 0):,}")
                    print(f"   🔍 Non-Branded Keywords: {branded_analysis.get('non_branded_count', 0):,}")
            
            if 'competitive_gaps' in results:
                quick_wins = results['competitive_gaps'].get('quick_wins', [])
                print(f"   ⚡ Quick Win Opportunities: {len(quick_wins)}")
        
        # Data quality summary
        print(f"\n📊 Data Quality Summary:")
        print(f"   ✅ Overall Quality Score: {overall_quality:.2f}")
        print(f"   📁 Datasets Processed: {validation_results['summary']['total_datasets']}")
        print(f"   📄 Total Rows: {validation_results['summary']['total_rows']:,}")
        
        logger.info("Enhanced SEO analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis Error: {e}")
        import traceback
        traceback.print_exc()

def generate_executive_summary(results, validation_results):
    """Generate executive summary with key insights"""
    
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
    
    return summary

def print_executive_summary(summary):
    """Print formatted executive summary"""
    
    print("📋 EXECUTIVE SUMMARY")
    print("=" * 40)
    print(f"Analysis Date: {summary['analysis_date']}")
    print(f"Data Quality: {summary['data_quality_score']:.2f}/1.00")
    print(f"Datasets: {summary['datasets_analyzed']}")
    print(f"Total Keywords: {summary.get('total_keywords_analyzed', 0):,}")
    
    print("\n🏢 LENOVO PERFORMANCE")
    print("-" * 25)
    print(f"Keywords Analyzed: {summary.get('lenovo_keywords', 0):,}")
    print(f"Average Position: {summary.get('average_position', 0):.1f}")
    print(f"Traffic Share: {summary.get('traffic_share', 0):.1f}%")
    print(f"Top 10 Rankings: {summary.get('top_10_rankings', 0):,}")
    print(f"Market Share: {summary.get('market_share', 0):.1f}%")
    
    print("\n🎯 ADVANCED INSIGHTS")
    print("-" * 20)
    print(f"SERP Coverage: {summary.get('serp_coverage', 0):.1f}%")
    print(f"Commercial Keywords: {summary.get('commercial_keywords', 0):,}")
    print(f"Branded Keywords: {summary.get('branded_keywords', 0):,}")
    
    print("\n⚡ OPPORTUNITIES")
    print("-" * 15)
    print(f"Position Gaps: {summary.get('position_gaps', 0)}")
    print(f"Quick Wins: {summary.get('quick_wins', 0)}")

if __name__ == "__main__":
    main()
