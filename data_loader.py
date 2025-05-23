"""
Simple SEO Data Loader
Just loads Excel files from date folders - no complexity
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

class SEODataLoader:
    """Simple, reliable data loader"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all SEO data from date folders"""
        
        data = {
            'lenovo': pd.DataFrame(),
            'dell': pd.DataFrame(), 
            'hp': pd.DataFrame(),
            'gap_keywords': pd.DataFrame()
        }
        
        # Fixed date folders
        date_folders = ["May-19-2025", "May-20-2025", "May-21-2025"]
        
        print("📁 Loading SEO data...")
        
        for date_folder in date_folders:
            folder_path = Path(f"data/{date_folder}")
            
            if not folder_path.exists():
                continue
                
            print(f"   📂 Processing {date_folder}")
            
            # Load files by pattern
            for file_path in folder_path.glob("*.xlsx"):
                filename = file_path.name.lower()
                
                try:
                    df = pd.read_excel(file_path)
                    df['source_date'] = date_folder
                    df['source_file'] = file_path.name
                    
                    # Categorize by filename
                    if 'lenovo' in filename and 'positions' in filename:
                        data['lenovo'] = pd.concat([data['lenovo'], df], ignore_index=True)
                        print(f"      ✅ Lenovo: +{len(df)} rows")
                        
                    elif 'dell' in filename and 'positions' in filename:
                        data['dell'] = pd.concat([data['dell'], df], ignore_index=True)
                        print(f"      ✅ Dell: +{len(df)} rows")
                        
                    elif 'hp' in filename and 'positions' in filename:
                        data['hp'] = pd.concat([data['hp'], df], ignore_index=True)
                        print(f"      ✅ HP: +{len(df)} rows")
                        
                    elif 'gap.keywords' in filename:
                        data['gap_keywords'] = pd.concat([data['gap_keywords'], df], ignore_index=True)
                        print(f"      ✅ Gap Keywords: +{len(df)} rows")
                        
                except Exception as e:
                    print(f"      ❌ Error loading {file_path.name}: {e}")
        
        # Summary
        print(f"\n📊 Data Summary:")
        for key, df in data.items():
            print(f"   {key}: {len(df):,} total rows")
            
        return data
