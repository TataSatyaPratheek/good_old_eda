"""
Simple SEO Analyzer
Core analysis functions without over-engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class SEOAnalyzer:
    """Simple SEO analysis"""
    
    def analyze_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run complete SEO analysis"""
        
        print("🔍 Running SEO analysis...")
        
        results = {
            'summary': self._create_summary(data),
            'competitive': self._competitive_analysis(data),
            'keyword_analysis': self._keyword_analysis(data),
            'opportunity_analysis': self._opportunity_analysis(data)
        }
        
        return results
    
    def _create_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create data summary"""
        
        summary = {}
        
        for company, df in data.items():
            if df.empty:
                continue
                
            summary[company] = {
                'total_keywords': len(df),
                'avg_position': df.get('Position', pd.Series()).mean(),
                'total_traffic': df.get('Traffic (%)', pd.Series()).sum(),
                'top_10_count': len(df[df.get('Position', 100) <= 10]) if 'Position' in df.columns else 0,
                'date_range': f"{df['source_date'].min()} to {df['source_date'].max()}"
            }
            
        return summary
    
    def _competitive_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Simple competitive analysis"""
        
        lenovo_data = data.get('lenovo', pd.DataFrame())
        dell_data = data.get('dell', pd.DataFrame())
        hp_data = data.get('hp', pd.DataFrame())
        
        # Traffic comparison
        traffic_comparison = {
            'lenovo': lenovo_data.get('Traffic (%)', pd.Series()).sum(),
            'dell': dell_data.get('Traffic (%)', pd.Series()).sum(),
            'hp': hp_data.get('Traffic (%)', pd.Series()).sum()
        }
        
        total_traffic = sum(traffic_comparison.values())
        market_share = {
            company: (traffic / total_traffic * 100) if total_traffic > 0 else 0
            for company, traffic in traffic_comparison.items()
        }
        
        # Keyword overlap
        keyword_overlap = {}
        if not lenovo_data.empty and 'Keyword' in lenovo_data.columns:
            lenovo_keywords = set(lenovo_data['Keyword'].str.lower())
            
            for competitor, comp_data in [('dell', dell_data), ('hp', hp_data)]:
                if not comp_data.empty and 'Keyword' in comp_data.columns:
                    comp_keywords = set(comp_data['Keyword'].str.lower())
                    overlap = len(lenovo_keywords.intersection(comp_keywords))
                    keyword_overlap[competitor] = overlap
        
        return {
            'traffic_comparison': traffic_comparison,
            'market_share': market_share,
            'keyword_overlap': keyword_overlap
        }
    
    def _keyword_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Keyword performance analysis"""
        
        lenovo_data = data.get('lenovo', pd.DataFrame())
        
        if lenovo_data.empty or 'Position' not in lenovo_data.columns:
            return {}
        
        # Position distribution
        position_ranges = {
            'top_3': len(lenovo_data[lenovo_data['Position'] <= 3]),
            'top_10': len(lenovo_data[lenovo_data['Position'] <= 10]),
            'top_20': len(lenovo_data[lenovo_data['Position'] <= 20]),
            'beyond_20': len(lenovo_data[lenovo_data['Position'] > 20])
        }
        
        # Best and worst performers
        best_keywords = lenovo_data.nsmallest(10, 'Position')[['Keyword', 'Position', 'Traffic (%)']].to_dict('records') if not lenovo_data.empty else []
        worst_keywords = lenovo_data.nlargest(10, 'Position')[['Keyword', 'Position', 'Traffic (%)']].to_dict('records') if not lenovo_data.empty else []
        
        return {
            'position_distribution': position_ranges,
            'best_keywords': best_keywords,
            'worst_keywords': worst_keywords,
            'avg_position': lenovo_data['Position'].mean()
        }
    
    def _opportunity_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Find opportunities"""
        
        gap_data = data.get('gap_keywords', pd.DataFrame())
        
        if gap_data.empty:
            return {}
        
        # High-value opportunities
        opportunities = []
        
        if 'Volume' in gap_data.columns and 'Keyword Difficulty' in gap_data.columns:
            # Filter for high volume, low difficulty
            good_opportunities = gap_data[
                (gap_data['Volume'] > 1000) & 
                (gap_data['Keyword Difficulty'] < 50)
            ]
            
            opportunities = good_opportunities.head(20)[['Keyword', 'Volume', 'Keyword Difficulty']].to_dict('records')
        
        return {
            'total_gap_keywords': len(gap_data),
            'high_value_opportunities': opportunities,
            'avg_difficulty': gap_data.get('Keyword Difficulty', pd.Series()).mean()
        }
