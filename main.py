"""
Example usage of the comprehensive pipeline system with date-based data structure
"""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

# Import functions/classes that are designed to be available even if __init__.py has issues
from pipelines import (
    create_pipeline_orchestrator,
    get_pipeline_system_info, validate_pipeline_system
)

def setup_data_environment():
    """Configure environment for date-based data discovery"""
    
    # Set environment variables for data loading
    os.environ['SEO_DATA_SEARCH_SUBDIRS'] = 'true'
    os.environ['SEO_DATA_DATE_FOLDERS'] = 'May-*-2025'
    
    # Add specific data directories to search path
    data_base = Path("data")
    date_dirs = []
    
    if data_base.exists():
        # Find all date-based subdirectories
        for subdir in data_base.iterdir():
            if subdir.is_dir() and subdir.name.startswith('May-') and '2025' in subdir.name:
                date_dirs.append(str(subdir))
        
        # Set comprehensive data directories
        all_dirs = ['data'] + date_dirs
        os.environ['SEO_DATA_DIRECTORIES'] = ';'.join(all_dirs)
        
        print(f"🔍 Configured data search in: {len(date_dirs)} date directories")
        for dir_path in date_dirs:
            files_count = len(list(Path(dir_path).glob("*.xlsx")))
            print(f"   📁 {dir_path}: {files_count} Excel files")
    
    # Configure file patterns to match your naming convention
    os.environ['SEO_FILE_PATTERNS'] = 'positions:*-organic.Positions-*.xlsx;competitors:*-organic.Competitors-*.xlsx;gap_keywords:gap.keywords*.xlsx'
    
    # Set data quality and processing options
    os.environ['SEO_DATA_QUALITY_THRESHOLD'] = '0.5'  # Lower threshold for testing
    os.environ['SEO_FORCE_DATA_RELOAD'] = 'true'
    os.environ['SEO_RECURSIVE_SEARCH'] = 'true'
    
    # Set analysis configuration through environment variables
    os.environ['SEO_COMPETITORS'] = 'dell,hp'
    os.environ['SEO_PRIMARY_DOMAIN'] = 'lenovo'
    os.environ['SEO_INCLUDE_DATE_FOLDERS'] = 'true'
    os.environ['SEO_RECURSIVE_FILE_SEARCH'] = 'true'

def create_data_access_links():
    """Create symbolic links in data/current for easy access"""
    
    data_dir = Path("data")
    current_dir = data_dir / "current"
    current_dir.mkdir(exist_ok=True)
    
    # Find all Excel files in date directories
    excel_files = []
    for date_dir in data_dir.glob("May-*-2025"):
        if date_dir.is_dir():
            excel_files.extend(date_dir.glob("*.xlsx"))
    
    print(f"📋 Found {len(excel_files)} Excel files to link")
    
    # Create links/copies in current directory
    for excel_file in excel_files:
        link_path = current_dir / excel_file.name
        
        if not link_path.exists():
            try:
                # Try creating symbolic link
                link_path.symlink_to(excel_file.resolve())
                print(f"🔗 Linked: {excel_file.name}")
            except OSError:
                # Fall back to copying
                import shutil
                shutil.copy2(excel_file, link_path)
                print(f"📄 Copied: {excel_file.name}")
    
    return str(current_dir)

async def main():
    """Example of running the complete SEO analysis system with date-based data"""
    
    print("🚀 Starting SEO Competitive Intelligence Analysis")
    print("=" * 60)
    
    # Setup data environment first
    setup_data_environment()
    
    # Create data access links
    current_data_dir = create_data_access_links()
    print(f"✅ Data accessible via: {current_data_dir}")
    
    # Validate pipeline system
    print("\n🔍 Validating pipeline system...")
    validation = validate_pipeline_system()
    print("Pipeline System Validation:", validation)
    
    # Get system info
    system_info = get_pipeline_system_info()
    print("Pipeline System Info:", system_info)
    
    if not validation.get('imports_successful', False):
        print("\n❌ Pipeline system failed to initialize its components due to import errors.")
        if 'import_errors' in validation and validation['import_errors']:
            print(f"Detected errors: {validation['import_errors']}")
        else:
            print("No specific import errors reported, but imports_successful is false.")
        
        print("Attempting to create orchestrator for detailed error diagnosis...")
        try:
            create_pipeline_orchestrator() 
        except ImportError as e_factory:
            print(f"Pipeline system initialization failed: {e_factory}")
        except Exception as e_other_factory:
            print(f"Pipeline system initialization failed with unexpected error: {e_other_factory}")
        return

    # Import ExecutionMode after validation
    try:
        from pipelines import ExecutionMode
    except ImportError:
        print("\n❌ Critical Error: 'ExecutionMode' could not be imported")
        return

    # Create orchestrator
    try:
        print("\n🏗️ Creating pipeline orchestrator...")
        orchestrator = create_pipeline_orchestrator()
        print("✅ Orchestrator created successfully")
    except ImportError as e:
        print(f"\n❌ Error: Failed to create pipeline orchestrator: {e}")
        return
    except Exception as e:
        print(f"\n❌ Unexpected error creating orchestrator: {e}")
        return
        
    # Define analysis parameters matching your data structure
    # Your data spans May 19-22, 2025
    start_date = datetime(2025, 5, 19)  # May 19, 2025
    end_date = datetime(2025, 5, 23)    # May 23, 2025 (current date)
    date_range = (start_date, end_date)
    
    print(f"\n📅 Analysis date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Execute complete analysis with correct parameters
    print("\n🔄 Starting comprehensive SEO competitive intelligence analysis...")
    print("This may take a few minutes...")
    
    try:        
        # Call with only the supported parameters
        result = await orchestrator.execute_complete_analysis(
            execution_mode=ExecutionMode.DEPENDENCY_BASED,
            date_range=date_range,
            pipeline_subset=None,  # Execute all pipelines
            force_refresh=True     # Force refresh to ensure data reload
        )
        
        print("\n🎉 Analysis completed successfully!")
        print("=" * 60)
        print("📊 Execution Summary:", result.execution_summary)
        print("⚡ Performance Metrics:", result.performance_metrics)
        print("💡 Recommendations:", result.recommendations)
        
        # Get final orchestration status
        status = orchestrator.get_orchestration_status()
        print("\n📈 Orchestration Final Status:", status)
        
        # Display results summary
        if hasattr(result, 'pipeline_results'):
            print("\n📋 Pipeline Results Summary:")
            for pipeline_name, pipeline_result in result.pipeline_results.items():
                print(f"   {pipeline_name}: {'✅ Success' if pipeline_result else '❌ Failed'}")
        
        # Show available reports
        reports_dir = Path("reports")
        if reports_dir.exists():
            report_files = list(reports_dir.glob("*.html")) + list(reports_dir.glob("*.xlsx"))
            if report_files:
                print(f"\n📄 Generated Reports ({len(report_files)} files):")
                for report in sorted(report_files):
                    print(f"   📋 {report.name}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed during execution: {str(e)}")
        print("\n🔍 Detailed error information:")
        import traceback
        traceback.print_exc()
        
        # Try to get partial results or status
        try:
            status = orchestrator.get_orchestration_status()
            print(f"\n📊 Partial orchestration status: {status}")
        except:
            print("Could not retrieve orchestration status")

if __name__ == "__main__":
    asyncio.run(main())
