"""
Simple SEO Analysis - Clean and maintainable
"""

from data_loader import SEODataLoader
from seo_analyzer import SEOAnalyzer  
from report_generator import ReportGenerator

def main():
    """Run complete SEO analysis"""
    
    print("🚀 SEO Competitive Intelligence Analysis")
    print("📅 Data Period: May 19-21, 2025")
    print("=" * 50)
    
    try:
        # Load data
        loader = SEODataLoader()
        data = loader.load_all_data()
        
        # Run analysis
        analyzer = SEOAnalyzer()
        results = analyzer.analyze_all(data)
        
        # Generate reports
        reporter = ReportGenerator()
        reports = reporter.generate_all_reports(results)
        
        # Print summary
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Reports generated:")
        for report_type, path in reports.items():
            print(f"   📄 {report_type.upper()}: {path}")
        
        # Quick stats
        summary = results.get('summary', {})
        if 'lenovo' in summary:
            lenovo = summary['lenovo']
            print(f"\n📈 Lenovo Quick Stats:")
            print(f"   Keywords: {lenovo.get('total_keywords', 0):,}")
            print(f"   Avg Position: {lenovo.get('avg_position', 0):.1f}")
            print(f"   Traffic Share: {lenovo.get('total_traffic', 0):.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
