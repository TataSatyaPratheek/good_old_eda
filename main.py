"""
Example usage of the comprehensive pipeline system
"""

import asyncio
from datetime import datetime, timedelta

# Import functions/classes that are designed to be available even if __init__.py has issues
from pipelines import ( # type: ignore
    create_pipeline_orchestrator,
    get_pipeline_system_info, validate_pipeline_system
)

async def main():
    """Example of running the complete SEO analysis system"""
    
    # Validate pipeline system
    validation = validate_pipeline_system()
    print("Pipeline System Validation:", validation)
    
    # Get system info
    system_info = get_pipeline_system_info()
    print("Pipeline System Info:", system_info)
    
    if not validation.get('imports_successful', False):
        print("\nPipeline system failed to initialize its components due to import errors.")
        if 'import_errors' in validation and validation['import_errors']:
            print(f"Detected errors from pipeline system initialization: {validation['import_errors']}")
        else:
            print("No specific import errors reported by validation, but imports_successful is false.")
        print("Attempting to create orchestrator to see detailed error (if any from fallback factory):")
        try:
            # This call is intended to trigger the informative error from the fallback factory in __init__.py
            create_pipeline_orchestrator() 
        except ImportError as e_factory:
            print(f"Pipeline system initialization failed: {e_factory}")
        except Exception as e_other_factory:
            print(f"Pipeline system initialization failed with an unexpected error: {e_other_factory}")
        return

    # If validation was successful, we can now try to import ExecutionMode
    try:
        from pipelines import ExecutionMode
    except ImportError:
        print("\nCritical Error: 'ExecutionMode' could not be imported even though pipeline system validation reported success.")
        print("This indicates an unexpected issue in 'pipelines/__init__.py' or its exports.")
        return

    # Create orchestrator
    try:
        orchestrator = create_pipeline_orchestrator()
    except ImportError as e: # Should not happen if validation was successful
        print(f"\nError: Failed to create pipeline orchestrator unexpectedly: {e}")
        return
        
    # Define analysis parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Last 7 days
    date_range = (start_date, end_date)
    
    # Execute complete analysis
    print("Starting comprehensive SEO competitive intelligence analysis...")
    
    try:        
        result = await orchestrator.execute_complete_analysis(
            execution_mode=ExecutionMode.DEPENDENCY_BASED,
            date_range=date_range,
            pipeline_subset=None,  # Execute all pipelines
            force_refresh=False
        )
        
        print("\nAnalysis completed successfully!")
        print("Execution Summary:", result.execution_summary)
        print("Performance Metrics:", result.performance_metrics)
        print("Recommendations:", result.recommendations)
        
        # Get orchestration status
        status = orchestrator.get_orchestration_status()
        print("Orchestration Final Status:", status)
        
    except Exception as e:
        print(f"\nAnalysis failed during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
