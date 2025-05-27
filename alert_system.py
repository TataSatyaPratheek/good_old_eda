"""
Alert System Module
Real-time monitoring and alerting for SEO performance changes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    TRAFFIC_DROP = "traffic_drop"
    RANKING_DROP = "ranking_drop"
    COMPETITIVE_THREAT = "competitive_threat"
    ALGORITHM_IMPACT = "algorithm_impact"
    OPPORTUNITY = "opportunity"
    TECHNICAL_ISSUE = "technical_issue"

@dataclass
class Alert:
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    affected_keywords: List[str]
    metrics: Dict[str, float]
    timestamp: datetime
    recommended_actions: List[str]
    auto_resolve: bool = False

class AlertSystem:
    """Real-time SEO monitoring and alerting system"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alert_rules = self._setup_alert_rules()
        self.active_alerts = []
        self.alert_history = []
        
    def monitor_and_alert(self, current_data: Dict[str, pd.DataFrame],
                        historical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Monitor SEO data and generate alerts
        """
        
        print("🚨 Monitoring for SEO Alerts...")
        
        monitoring_results = {
            'new_alerts': [],
            'resolved_alerts': [],
            'active_alerts': [],
            'alert_summary': {},
            'monitoring_status': 'healthy'
        }
        
        # Run alert detection
        new_alerts = self._detect_alerts(current_data, historical_data)
        monitoring_results['new_alerts'] = new_alerts
        
        # Update active alerts
        self._update_active_alerts(new_alerts)
        monitoring_results['active_alerts'] = self.active_alerts
        
        # Generate alert summary
        monitoring_results['alert_summary'] = self._generate_alert_summary()
        
        # Determine overall monitoring status
        monitoring_results['monitoring_status'] = self._determine_monitoring_status()
        
        return monitoring_results

    def _setup_alert_rules(self) -> Dict[str, Callable]:
        """Setup alert detection rules"""
        
        return {
            'traffic_drop_detection': self._detect_traffic_drops,
            'ranking_drop_detection': self._detect_ranking_drops,
            'competitive_threat_detection': self._detect_competitive_threats,
            'opportunity_detection': self._detect_opportunities,
            'anomaly_detection': self._detect_anomalies,
            'serp_feature_loss_detection': self._detect_serp_feature_loss,
            'keyword_cannibalization_detection': self._detect_keyword_cannibalization
        }

    def _detect_alerts(self, current_data: Dict[str, pd.DataFrame],
                    historical_data: Dict[str, pd.DataFrame] = None) -> List[Alert]:
        """Detect and generate alerts"""
        
        alerts = []
        
        # Run each alert rule
        for rule_name, rule_function in self.alert_rules.items():
            try:
                rule_alerts = rule_function(current_data, historical_data)
                alerts.extend(rule_alerts)
            except Exception as e:
                self.logger.error(f"Error in alert rule {rule_name}: {e}")
        
        # Sort alerts by severity and timestamp
        alerts.sort(key=lambda x: (x.severity.value, x.timestamp), reverse=True)
        
        return alerts

    def _detect_traffic_drops(self, current_data: Dict[str, pd.DataFrame],
                            historical_data: Dict[str, pd.DataFrame] = None) -> List[Alert]:
        """Detect traffic drop alerts"""
        
        alerts = []
        
        if not historical_data:
            return alerts
        
        for brand, current_df in current_data.items():
            if brand == 'gap_keywords' or current_df.empty:
                continue
            
            historical_df = historical_data.get(brand, pd.DataFrame())
            if historical_df.empty:
                continue
            
            # Compare traffic changes
            traffic_changes = self._calculate_traffic_changes(current_df, historical_df)
            
            # Detect significant drops
            significant_drops = [
                change for change in traffic_changes
                if change['traffic_change_percent'] < -30  # 30% drop threshold
            ]
            
            if significant_drops:
                # Categorize by severity
                critical_drops = [d for d in significant_drops if d['traffic_change_percent'] < -60]
                high_drops = [d for d in significant_drops if -60 <= d['traffic_change_percent'] < -45]
                medium_drops = [d for d in significant_drops if -45 <= d['traffic_change_percent'] < -30]
                
                # Create alerts based on severity
                if critical_drops:
                    alerts.append(self._create_traffic_drop_alert(
                        brand, critical_drops, AlertSeverity.CRITICAL
                    ))
                elif high_drops:
                    alerts.append(self._create_traffic_drop_alert(
                        brand, high_drops, AlertSeverity.HIGH
                    ))
                elif medium_drops:
                    alerts.append(self._create_traffic_drop_alert(
                        brand, medium_drops, AlertSeverity.MEDIUM
                    ))
        
        return alerts

    def _calculate_traffic_changes(self, current_df: pd.DataFrame, 
                                historical_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate traffic changes between datasets"""
        
        if 'Keyword' not in current_df.columns or 'Keyword' not in historical_df.columns:
            return []
        
        # Merge dataframes
        merged = pd.merge(
            current_df[['Keyword', 'Traffic (%)']],
            historical_df[['Keyword', 'Traffic (%)']],
            on='Keyword',
            suffixes=('_current', '_historical')
        )
        
        changes = []
        for _, row in merged.iterrows():
            current_traffic = row['Traffic (%)_current']
            historical_traffic = row['Traffic (%)_historical']
            
            if historical_traffic > 0:
                change_percent = ((current_traffic - historical_traffic) / historical_traffic) * 100
                changes.append({
                    'keyword': row['Keyword'],
                    'current_traffic': current_traffic,
                    'historical_traffic': historical_traffic,
                    'traffic_change_percent': change_percent,
                    'traffic_change_absolute': current_traffic - historical_traffic
                })
        
        return changes

    def _create_traffic_drop_alert(self, brand: str, drops: List[Dict[str, Any]], 
                                severity: AlertSeverity) -> Alert:
        """Create traffic drop alert"""
        
        affected_keywords = [drop['keyword'] for drop in drops]
        avg_drop = np.mean([drop['traffic_change_percent'] for drop in drops])
        total_traffic_lost = sum([abs(drop['traffic_change_absolute']) for drop in drops])
        
        alert_id = f"traffic_drop_{brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.TRAFFIC_DROP,
            severity=severity,
            title=f"Traffic Drop Alert - {brand.title()}",
            description=f"{len(drops)} keywords showing {avg_drop:.1f}% average traffic drop. Total traffic lost: {total_traffic_lost:.2f}%",
            affected_keywords=affected_keywords,
            metrics={
                'avg_drop_percent': avg_drop,
                'total_traffic_lost': total_traffic_lost,
                'affected_keyword_count': len(drops)
            },
            timestamp=datetime.now(),
            recommended_actions=[
                f"Immediate audit of affected {len(drops)} keywords",
                "Check for technical issues or algorithm impacts",
                "Analyze competitor movements in affected keyword space",
                "Review content quality and relevance"
            ]
        )

    def _detect_ranking_drops(self, current_data: Dict[str, pd.DataFrame],
                            historical_data: Dict[str, pd.DataFrame] = None) -> List[Alert]:
        """Detect ranking drop alerts"""
        
        alerts = []
        
        if not historical_data:
            return alerts
        
        for brand, current_df in current_data.items():
            if brand == 'gap_keywords' or current_df.empty:
                continue
            
            historical_df = historical_data.get(brand, pd.DataFrame())
            if historical_df.empty:
                continue
            
            # Calculate ranking changes
            ranking_changes = self._calculate_ranking_changes(current_df, historical_df)
            
            # Detect significant ranking drops
            significant_drops = [
                change for change in ranking_changes
                if change['position_change'] >= 10  # 10+ position drop
            ]
            
            if significant_drops:
                severity = self._determine_ranking_drop_severity(significant_drops)
                alert = self._create_ranking_drop_alert(brand, significant_drops, severity)
                alerts.append(alert)
        
        return alerts

    def _calculate_ranking_changes(self, current_df: pd.DataFrame, 
                                historical_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate ranking changes between datasets"""
        
        if 'Keyword' not in current_df.columns or 'Keyword' not in historical_df.columns:
            return []
        
        # Merge dataframes
        merged = pd.merge(
            current_df[['Keyword', 'Position', 'Traffic (%)']],
            historical_df[['Keyword', 'Position', 'Traffic (%)']],
            on='Keyword',
            suffixes=('_current', '_historical')
        )
        
        changes = []
        for _, row in merged.iterrows():
            current_position = row['Position_current']
            historical_position = row['Position_historical']
            position_change = current_position - historical_position
            
            changes.append({
                'keyword': row['Keyword'],
                'current_position': current_position,
                'historical_position': historical_position,
                'position_change': position_change,
                'current_traffic': row['Traffic (%)_current'],
                'historical_traffic': row['Traffic (%)_historical']
            })
        
        return changes

    def _determine_ranking_drop_severity(self, drops: List[Dict[str, Any]]) -> AlertSeverity:
        """Determine severity of ranking drops"""
        
        avg_drop = np.mean([drop['position_change'] for drop in drops])
        max_drop = max([drop['position_change'] for drop in drops])
        
        # Check for page 1 to page 2+ drops
        page_drops = [d for d in drops if d['historical_position'] <= 10 and d['current_position'] > 10]
        
        if max_drop >= 20 or len(page_drops) >= 3:
            return AlertSeverity.CRITICAL
        elif avg_drop >= 15 or len(page_drops) >= 1:
            return AlertSeverity.HIGH
        else:
            return AlertSeverity.MEDIUM

    def _create_ranking_drop_alert(self, brand: str, drops: List[Dict[str, Any]], 
                                severity: AlertSeverity) -> Alert:
        """Create ranking drop alert"""
        
        affected_keywords = [drop['keyword'] for drop in drops]
        avg_drop = np.mean([drop['position_change'] for drop in drops])
        page_drops = [d for d in drops if d['historical_position'] <= 10 and d['current_position'] > 10]
        
        alert_id = f"ranking_drop_{brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.RANKING_DROP,
            severity=severity,
            title=f"Ranking Drop Alert - {brand.title()}",
            description=f"{len(drops)} keywords dropped {avg_drop:.1f} positions on average. {len(page_drops)} fell from page 1.",
            affected_keywords=affected_keywords,
            metrics={
                'avg_position_drop': avg_drop,
                'affected_keyword_count': len(drops),
                'page_1_drops': len(page_drops)
            },
            timestamp=datetime.now(),
            recommended_actions=[
                "Immediate ranking recovery plan for affected keywords",
                "Technical SEO audit to identify issues",
                "Content optimization for dropped keywords",
                "Backlink analysis and acquisition strategy"
            ]
        )

    def _detect_competitive_threats(self, current_data: Dict[str, pd.DataFrame],
                                historical_data: Dict[str, pd.DataFrame] = None) -> List[Alert]:
        """Detect competitive threat alerts"""
        
        alerts = []
        
        lenovo_df = current_data.get('lenovo', pd.DataFrame())
        if lenovo_df.empty:
            return alerts
        
        # Analyze competitor movements
        for competitor in ['dell', 'hp']:
            competitor_df = current_data.get(competitor, pd.DataFrame())
            if competitor_df.empty:
                continue
            
            threats = self._identify_competitive_threats(lenovo_df, competitor_df, competitor)
            
            if threats:
                severity = self._determine_threat_severity(threats)
                alert = self._create_competitive_threat_alert(competitor, threats, severity)
                alerts.append(alert)
        
        return alerts

    def _identify_competitive_threats(self, lenovo_df: pd.DataFrame, 
                                    competitor_df: pd.DataFrame, competitor: str) -> List[Dict[str, Any]]:
        """Identify competitive threats"""
        
        if 'Keyword' not in lenovo_df.columns or 'Keyword' not in competitor_df.columns:
            return []
        
        threats = []
        
        # Find overlapping keywords where competitor is outperforming
        merged = pd.merge(
            lenovo_df[['Keyword', 'Position', 'Traffic (%)']],
            competitor_df[['Keyword', 'Position', 'Traffic (%)']],
            on='Keyword',
            suffixes=('_lenovo', '_competitor')
        )
        
        for _, row in merged.iterrows():
            lenovo_position = row['Position_lenovo']
            competitor_position = row['Position_competitor']
            
            # Threat conditions
            if (competitor_position < lenovo_position and 
                competitor_position <= 10 and 
                lenovo_position > 10):
                
                threat_level = 'critical' if competitor_position <= 3 else 'high'
                
                threats.append({
                    'keyword': row['Keyword'],
                    'lenovo_position': lenovo_position,
                    'competitor_position': competitor_position,
                    'position_gap': lenovo_position - competitor_position,
                    'lenovo_traffic': row['Traffic (%)_lenovo'],
                    'competitor_traffic': row['Traffic (%)_competitor'],
                    'threat_level': threat_level
                })
        
        return threats

    def _determine_threat_severity(self, threats: List[Dict[str, Any]]) -> AlertSeverity:
        """Determine severity of competitive threats"""
        
        critical_threats = [t for t in threats if t['threat_level'] == 'critical']
        high_threats = [t for t in threats if t['threat_level'] == 'high']
        
        if len(critical_threats) >= 3:
            return AlertSeverity.CRITICAL
        elif len(critical_threats) >= 1 or len(high_threats) >= 5:
            return AlertSeverity.HIGH
        else:
            return AlertSeverity.MEDIUM

    def _create_competitive_threat_alert(self, competitor: str, threats: List[Dict[str, Any]], 
                                    severity: AlertSeverity) -> Alert:
        """Create competitive threat alert"""
        
        affected_keywords = [threat['keyword'] for threat in threats]
        avg_gap = np.mean([threat['position_gap'] for threat in threats])
        critical_threats = [t for t in threats if t['threat_level'] == 'critical']
        
        alert_id = f"competitive_threat_{competitor}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.COMPETITIVE_THREAT,
            severity=severity,
            title=f"Competitive Threat - {competitor.title()}",
            description=f"{competitor.title()} outranking on {len(threats)} keywords with {avg_gap:.1f} average position gap. {len(critical_threats)} critical threats.",
            affected_keywords=affected_keywords,
            metrics={
                'threat_count': len(threats),
                'avg_position_gap': avg_gap,
                'critical_threats': len(critical_threats)
            },
            timestamp=datetime.now(),
            recommended_actions=[
                f"Competitive analysis of {competitor.title()}'s strategy",
                "Develop counter-offensive for high-value keywords",
                "Content improvement for threatened keywords",
                "Monitor competitor's new content and link building"
            ]
        )

    def _detect_opportunities(self, current_data: Dict[str, pd.DataFrame],
                            historical_data: Dict[str, pd.DataFrame] = None) -> List[Alert]:
        """Detect opportunity alerts"""
        
        alerts = []
        
        lenovo_df = current_data.get('lenovo', pd.DataFrame())
        if lenovo_df.empty:
            return alerts
        
        # Detect quick win opportunities
        quick_wins = self._identify_quick_wins(lenovo_df)
        
        if quick_wins:
            alert = self._create_opportunity_alert(quick_wins)
            alerts.append(alert)
        
        return alerts

    def _identify_quick_wins(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify quick win opportunities"""
        
        if 'Position' not in df.columns or 'Traffic (%)' not in df.columns:
            return []
        
        # Page 2 keywords with decent traffic potential
        quick_wins = []
        page_2_keywords = df[
            (df['Position'] >= 11) & 
            (df['Position'] <= 20) & 
            (df['Traffic (%)'] > 0.3)
        ]
        
        for _, row in page_2_keywords.iterrows():
            opportunity_score = self._calculate_opportunity_score(row)
            quick_wins.append({
                'keyword': row['Keyword'],
                'current_position': row['Position'],
                'current_traffic': row['Traffic (%)'],
                'opportunity_score': opportunity_score,
                'estimated_traffic_gain': self._estimate_traffic_gain(row),
                'optimization_difficulty': 'medium',
                'expected_timeline': '2-4 weeks'
            })
        
        return quick_wins
    
    def _calculate_opportunity_score(self, row: pd.Series) -> float:
        """Calculate opportunity score for a keyword"""
        
        position = row['Position']
        traffic = row['Traffic (%)']
        
        # Higher score for better positions and higher traffic
        position_score = max(0, (25 - position) / 25 * 50)
        traffic_score = min(traffic * 10, 50)
        
        return position_score + traffic_score
    
    def _estimate_traffic_gain(self, row: pd.Series) -> float:
        """Estimate potential traffic gain if keyword reaches page 1"""
        
        current_position = row['Position']
        current_traffic = row['Traffic (%)']
        
        # CTR estimates for different positions
        ctr_estimates = {
            1: 0.284, 2: 0.147, 3: 0.103, 4: 0.073, 5: 0.053,
            6: 0.040, 7: 0.031, 8: 0.025, 9: 0.020, 10: 0.017
        }
        
        # Estimate current CTR for page 2
        current_ctr = 0.005 if current_position > 10 else ctr_estimates.get(int(current_position), 0.017)
        
        # Target CTR for position 8-10
        target_ctr = ctr_estimates.get(10, 0.017)
        
        if current_ctr == 0:
            return 0
        
        traffic_multiplier = target_ctr / current_ctr
        return current_traffic * (traffic_multiplier - 1)  # Additional traffic gain
    
    def _create_opportunity_alert(self, quick_wins: List[Dict[str, Any]]) -> Alert:
        """Create opportunity alert"""
        
        high_value_wins = [qw for qw in quick_wins if qw['opportunity_score'] > 60]
        total_traffic_potential = sum([qw['estimated_traffic_gain'] for qw in quick_wins])
        
        alert_id = f"opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.OPPORTUNITY,
            severity=AlertSeverity.MEDIUM,
            title="Quick Win Opportunities Detected",
            description=f"{len(quick_wins)} quick win opportunities identified. {len(high_value_wins)} high-value opportunities with {total_traffic_potential:.1f}% potential traffic gain.",
            affected_keywords=[qw['keyword'] for qw in quick_wins[:10]],
            metrics={
                'total_opportunities': len(quick_wins),
                'high_value_opportunities': len(high_value_wins),
                'total_traffic_potential': total_traffic_potential
            },
            timestamp=datetime.now(),
            recommended_actions=[
                "Prioritize high-value quick win opportunities",
                "Optimize content for page 2 keywords",
                "Improve technical SEO for targeted keywords",
                "Monitor progress weekly"
            ]
        )
    
    def _detect_anomalies(self, current_data: Dict[str, pd.DataFrame],
                        historical_data: Dict[str, pd.DataFrame] = None) -> List[Alert]:
        """Detect anomalies in SEO data"""
        
        alerts = []
        
        for brand, current_df in current_data.items():
            if brand == 'gap_keywords' or current_df.empty:
                continue
            
            # Detect position anomalies
            position_anomalies = self._detect_position_anomalies(current_df)
            if position_anomalies:
                alert = self._create_anomaly_alert(brand, position_anomalies, 'position')
                alerts.append(alert)
            
            # Detect traffic anomalies
            traffic_anomalies = self._detect_traffic_anomalies(current_df)
            if traffic_anomalies:
                alert = self._create_anomaly_alert(brand, traffic_anomalies, 'traffic')
                alerts.append(alert)
        
        return alerts
    
    def _detect_position_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect position anomalies using statistical methods"""
        
        if 'Position' not in df.columns or len(df) < 10:
            return []
        
        positions = df['Position'].values
        
        # Calculate z-scores
        mean_position = np.mean(positions)
        std_position = np.std(positions)
        
        if std_position == 0:
            return []
        
        z_scores = np.abs((positions - mean_position) / std_position)
        
        # Identify anomalies (z-score > 3)
        anomaly_indices = np.where(z_scores > 3)[0]
        
        anomalies = []
        for idx in anomaly_indices:
            anomalies.append({
                'keyword': df.iloc[idx]['Keyword'],
                'position': df.iloc[idx]['Position'],
                'z_score': z_scores[idx],
                'anomaly_type': 'statistical_outlier'
            })
        
        return anomalies
    
    def _detect_traffic_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect traffic anomalies"""
        
        if 'Traffic (%)' not in df.columns or len(df) < 10:
            return []
        
        traffic = df['Traffic (%)'].values
        
        # Use IQR method for traffic anomalies
        q75, q25 = np.percentile(traffic, [75, 25])
        iqr = q75 - q25
        
        if iqr == 0:
            return []
        
        lower_bound = q25 - (3 * iqr)
        upper_bound = q75 + (3 * iqr)
        
        anomalies = []
        for idx, row in df.iterrows():
            traffic_val = row['Traffic (%)']
            if traffic_val < lower_bound or traffic_val > upper_bound:
                anomalies.append({
                    'keyword': row['Keyword'],
                    'traffic': traffic_val,
                    'expected_range': f"{q25:.2f} - {q75:.2f}",
                    'anomaly_type': 'traffic_outlier'
                })
        
        return anomalies
    
    def _create_anomaly_alert(self, brand: str, anomalies: List[Dict[str, Any]], 
                            anomaly_type: str) -> Alert:
        """Create anomaly alert"""
        
        alert_id = f"anomaly_{anomaly_type}_{brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.TECHNICAL_ISSUE,
            severity=AlertSeverity.MEDIUM,
            title=f"{anomaly_type.title()} Anomalies - {brand.title()}",
            description=f"{len(anomalies)} {anomaly_type} anomalies detected for {brand}",
            affected_keywords=[anomaly['keyword'] for anomaly in anomalies[:10]],
            metrics={
                'anomaly_count': len(anomalies),
                'anomaly_type': anomaly_type
            },
            timestamp=datetime.now(),
            recommended_actions=[
                f"Investigate {anomaly_type} anomalies",
                "Check for data quality issues",
                "Verify tracking implementation",
                "Review recent changes"
            ]
        )
    
    def _detect_serp_feature_loss(self, current_data: Dict[str, pd.DataFrame],
                                historical_data: Dict[str, pd.DataFrame] = None) -> List[Alert]:
        """Detect SERP feature losses using real data comparison"""
        
        alerts = []
        
        if not historical_data:
            return alerts
        
        for brand, current_df in current_data.items():
            if brand == 'gap_keywords' or current_df.empty:
                continue
            
            # Get historical data for comparison
            historical_df = historical_data.get(brand, pd.DataFrame())
            if historical_df.empty:
                continue
            
            # Detect actual SERP feature losses
            feature_losses = self._detect_actual_serp_feature_loss(current_df, historical_df)
            
            if feature_losses:
                alert = self._create_serp_loss_alert(brand, feature_losses)
                alerts.append(alert)
        
        return alerts
    
    def _detect_actual_serp_feature_loss(self, current_df: pd.DataFrame, 
                                       historical_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect actual SERP feature losses using real data patterns"""
        
        if ('Keyword' not in current_df.columns or 'Keyword' not in historical_df.columns or
            'Position' not in current_df.columns or 'Position' not in historical_df.columns or
            'Traffic (%)' not in current_df.columns or 'Traffic (%)' not in historical_df.columns):
            return []
        
        # Merge current and historical data
        merged = pd.merge(
            current_df[['Keyword', 'Position', 'Traffic (%)']],
            historical_df[['Keyword', 'Position', 'Traffic (%)']],
            on='Keyword',
            suffixes=('_current', '_historical'),
            how='inner'
        )
        
        feature_losses = []
        
        for _, row in merged.iterrows():
            current_position = row['Position_current']
            historical_position = row['Position_historical']
            current_traffic = row['Traffic (%)_current']
            historical_traffic = row['Traffic (%)_historical']
            
            # Detect potential SERP feature loss patterns:
            # 1. Position stayed the same or improved slightly, but traffic dropped significantly
            # 2. Position is in top 5 but traffic is disproportionately low
            # 3. Large traffic drop without corresponding position drop
            
            position_change = current_position - historical_position
            
            if historical_traffic > 0:
                traffic_change_percent = ((current_traffic - historical_traffic) / historical_traffic) * 100
            else:
                traffic_change_percent = 0
            
            # Criteria for suspected SERP feature loss:
            suspicious_patterns = []
            
            # Pattern 1: Stable/improved position but significant traffic drop
            if (position_change <= 2 and  # Position stable or improved slightly
                traffic_change_percent < -40 and  # Traffic dropped >40%
                current_position <= 5 and  # Still in top 5
                historical_traffic > 0.5):  # Had meaningful traffic before
                
                suspicious_patterns.append("stable_position_traffic_drop")
            
            # Pattern 2: Top position but unusually low traffic (compared to expected CTR)
            if (current_position <= 3 and  # Top 3 position
                current_traffic < 0.3 and  # Very low traffic
                self._is_traffic_below_expected_ctr(current_position, current_traffic)):
                
                suspicious_patterns.append("low_traffic_for_top_position")
            
            # Pattern 3: Moderate position change but disproportionate traffic loss
            if (abs(position_change) <= 5 and  # Small position change
                traffic_change_percent < -50 and  # Large traffic drop
                historical_traffic > 1.0):  # Had significant traffic
                
                suspicious_patterns.append("disproportionate_traffic_loss")
            
            if suspicious_patterns:
                # Determine most likely lost feature based on patterns
                suspected_feature = self._determine_suspected_feature(
                    current_position, historical_position, 
                    current_traffic, historical_traffic,
                    suspicious_patterns
                )
                
                feature_losses.append({
                    'keyword': row['Keyword'],
                    'current_position': current_position,
                    'historical_position': historical_position,
                    'current_traffic': current_traffic,
                    'historical_traffic': historical_traffic,
                    'traffic_change_percent': traffic_change_percent,
                    'position_change': position_change,
                    'suspected_feature': suspected_feature,
                    'confidence_score': self._calculate_loss_confidence(
                        suspicious_patterns, traffic_change_percent, position_change
                    ),
                    'suspicious_patterns': suspicious_patterns
                })
        
        # Sort by confidence score and return top candidates
        feature_losses.sort(key=lambda x: x['confidence_score'], reverse=True)
        return feature_losses[:10]  # Top 10 most confident detections

    def _is_traffic_below_expected_ctr(self, position: float, actual_traffic: float) -> bool:
        """Check if traffic is significantly below expected CTR for position"""
        
        # Expected CTR ranges for positions (conservative estimates)
        expected_ctr_ranges = {
            1: (0.20, 0.35),   # Position 1: 20-35%
            2: (0.10, 0.20),   # Position 2: 10-20%
            3: (0.07, 0.15),   # Position 3: 7-15%
            4: (0.05, 0.10),   # Position 4: 5-10%
            5: (0.03, 0.08)    # Position 5: 3-8%
        }
        
        pos_int = int(position)
        if pos_int in expected_ctr_ranges:
            min_expected, max_expected = expected_ctr_ranges[pos_int]
            # Convert traffic percentage to decimal for comparison
            actual_ctr = actual_traffic / 100
            
            # Consider it suspicious if actual CTR is less than 50% of minimum expected
            return actual_ctr < (min_expected * 0.5)
        
        return False

    def _determine_suspected_feature(self, current_pos: float, historical_pos: float,
                                   current_traffic: float, historical_traffic: float,
                                   patterns: List[str]) -> str:
        """Determine what SERP feature was likely lost"""
        
        traffic_loss_percent = 0
        if historical_traffic > 0:
            traffic_loss_percent = ((historical_traffic - current_traffic) / historical_traffic) * 100
        
        # Featured Snippet - usually positions 1-3, significant traffic impact
        if (current_pos <= 3 and 
            traffic_loss_percent > 50 and
            "stable_position_traffic_drop" in patterns):
            return "Featured Snippet"
        
        # Rich Results/Schema - any position, moderate traffic impact
        elif (traffic_loss_percent > 30 and
              "disproportionate_traffic_loss" in patterns):
            return "Rich Results/Schema Markup"
        
        # Image Pack - usually shows in top 5
        elif (current_pos <= 5 and
              "low_traffic_for_top_position" in patterns):
            return "Image Pack/Visual Results"
        
        # Local Pack - typically position 1-3
        elif (current_pos <= 3 and
              traffic_loss_percent > 40):
            return "Local Pack/Map Results"
        
        # Knowledge Panel - usually position 1
        elif (current_pos == 1 and
              traffic_loss_percent > 60):
            return "Knowledge Panel"
        
        # Video Results
        elif (current_pos <= 5 and
              traffic_loss_percent > 35):
            return "Video Results"
        
        # Generic SERP feature
        else:
            return "SERP Feature (Unknown Type)"

    def _calculate_loss_confidence(self, patterns: List[str], traffic_change: float, 
                                 position_change: float) -> float:
        """Calculate confidence score for SERP feature loss detection"""
        
        confidence = 0.0
        
        # Base confidence from traffic loss severity
        if traffic_change < -70:
            confidence += 40
        elif traffic_change < -50:
            confidence += 30
        elif traffic_change < -30:
            confidence += 20
        
        # Confidence from position stability (more stable = higher confidence)
        if abs(position_change) <= 1:
            confidence += 30
        elif abs(position_change) <= 3:
            confidence += 20
        elif abs(position_change) <= 5:
            confidence += 10
        
        # Confidence from pattern combinations
        if "stable_position_traffic_drop" in patterns:
            confidence += 20
        if "low_traffic_for_top_position" in patterns:
            confidence += 15
        if "disproportionate_traffic_loss" in patterns:
            confidence += 10
        
        # Bonus for multiple patterns
        if len(patterns) > 1:
            confidence += 10
        
        return min(confidence, 100)  # Cap at 100

    def _create_serp_loss_alert(self, brand: str, losses: List[Dict[str, Any]]) -> Alert:
        """Create SERP feature loss alert"""
        
        alert_id = f"serp_loss_{brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Summarize losses for the alert description
        avg_confidence = np.mean([loss.get('confidence_score', 0) for loss in losses])
        top_suspected_feature = losses[0]['suspected_feature'] if losses else "Unknown"

        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.TECHNICAL_ISSUE, # Or a more specific AlertType if created
            severity=AlertSeverity.HIGH, # Default, could be dynamic based on confidence/impact
            title=f"Suspected SERP Feature Loss - {brand.title()}",
            description=f"{len(losses)} keywords showing signs of SERP feature loss (e.g., {top_suspected_feature}). Avg confidence: {avg_confidence:.1f}%.",
            affected_keywords=[loss['keyword'] for loss in losses],
            metrics={
                'suspected_losses_count': len(losses),
                'average_confidence_score': avg_confidence,
                'top_suspected_feature_example': top_suspected_feature,
                'details_per_keyword': losses # Storing detailed loss info
            },
            timestamp=datetime.now(),
            recommended_actions=[
                "Audit SERP features for affected keywords immediately.",
                "Verify schema markup and structured data implementation.",
                "Analyze content quality and relevance for lost features (e.g., Featured Snippets).",
                "Monitor SERP feature recovery efforts closely."
            ]
        )

    def _create_serp_loss_alert(self, brand: str, losses: List[Dict[str, Any]]) -> Alert:
        """Create SERP feature loss alert"""
        
        alert_id = f"serp_loss_{brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.TECHNICAL_ISSUE,
            severity=AlertSeverity.HIGH,
            title=f"Suspected SERP Feature Loss - {brand.title()}",
            description=f"{len(losses)} keywords showing signs of SERP feature loss",
            affected_keywords=[loss['keyword'] for loss in losses],
            metrics={
                'suspected_losses': len(losses)
            },
            timestamp=datetime.now(),
            recommended_actions=[
                "Audit SERP features for affected keywords",
                "Optimize content for featured snippets",
                "Check schema markup implementation",
                "Monitor SERP feature recovery"
            ]
        )
    
    def _detect_keyword_cannibalization(self, current_data: Dict[str, pd.DataFrame],
                                    historical_data: Dict[str, pd.DataFrame] = None) -> List[Alert]:
        """Detect keyword cannibalization issues"""
        
        alerts = []
        
        for brand, current_df in current_data.items():
            if brand == 'gap_keywords' or current_df.empty:
                continue
            
            cannibalization_issues = self._identify_cannibalization(current_df)
            
            if cannibalization_issues:
                alert = self._create_cannibalization_alert(brand, cannibalization_issues)
                alerts.append(alert)
        
        return alerts
    
    def _identify_cannibalization(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify potential keyword cannibalization"""
        
        if 'Keyword' not in df.columns:
            return []
        
        cannibalization_issues = []
        
        # Group similar keywords
        keyword_groups = {}
        for _, row in df.iterrows():
            keyword = str(row['Keyword']).lower() # Ensure keyword is a string
            # Create groups based on first 3 words
            key_words = keyword.split()[:3]
            group_key = ' '.join(key_words)
            
            if group_key not in keyword_groups:
                keyword_groups[group_key] = []
            
            keyword_groups[group_key].append({
                'keyword': row['Keyword'],
                'position': row.get('Position', 0),
                'traffic': row.get('Traffic (%)', 0)
            })
        
        # Check for cannibalization in groups with multiple keywords
        for group_key, keywords in keyword_groups.items():
            if len(keywords) > 1:
                positions = [kw['position'] for kw in keywords]
                if max(positions) - min(positions) > 10:  # Large position spread
                    cannibalization_issues.append({
                        'keyword_group': group_key,
                        'affected_keywords': keywords,
                        'position_spread': max(positions) - min(positions),
                        'total_traffic': sum(kw['traffic'] for kw in keywords)
                    })
        
        return cannibalization_issues
    
    def _create_cannibalization_alert(self, brand: str, issues: List[Dict[str, Any]]) -> Alert:
        """Create keyword cannibalization alert"""
        
        alert_id = f"cannibalization_{brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        affected_keywords = []
        for issue in issues:
            affected_keywords.extend([kw['keyword'] for kw in issue['affected_keywords']])
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.TECHNICAL_ISSUE,
            severity=AlertSeverity.MEDIUM,
            title=f"Keyword Cannibalization - {brand.title()}",
            description=f"{len(issues)} potential cannibalization issues affecting {len(affected_keywords)} keywords",
            affected_keywords=affected_keywords[:10],
            metrics={
                'cannibalization_groups': len(issues),
                'affected_keywords': len(affected_keywords)
            },
            timestamp=datetime.now(),
            recommended_actions=[
                "Audit content for overlapping keywords",
                "Consolidate or differentiate competing pages",
                "Implement proper internal linking strategy",
                "Monitor keyword performance changes"
            ]
        )
    
    def _update_active_alerts(self, new_alerts: List[Alert]) -> None:
        """Update active alerts list"""
        
        # Add new alerts
        self.active_alerts.extend(new_alerts)
        
        # Remove resolved alerts (based on auto_resolve flag and age)
        current_time = datetime.now()
        self.active_alerts = [
            alert for alert in self.active_alerts
            if not self._should_resolve_alert(alert, current_time)
        ]
        
        # Move resolved alerts to history
        resolved_alerts = [
            alert for alert in self.active_alerts
            if self._should_resolve_alert(alert, current_time)
        ]
        self.alert_history.extend(resolved_alerts)
    
    def _should_resolve_alert(self, alert: Alert, current_time: datetime) -> bool:
        """Determine if alert should be auto-resolved"""
        
        # Auto-resolve alerts older than 7 days if marked for auto-resolve
        if alert.auto_resolve:
            alert_age = current_time - alert.timestamp
            return alert_age.days > 7
        
        # Keep critical alerts active longer
        if alert.severity == AlertSeverity.CRITICAL:
            alert_age = current_time - alert.timestamp
            return alert_age.days > 14
        
        return False
    
    def _generate_alert_summary(self) -> Dict[str, Any]:
        """Generate alert summary"""
        
        summary = {
            'active_alerts_count': len(self.active_alerts),
            'alerts_by_severity': {},
            'alerts_by_type': {},
            'most_affected_keywords': [],
            'recent_alert_trend': 'stable'
        }
        
        # Count by severity
        for severity in AlertSeverity:
            count = len([alert for alert in self.active_alerts if alert.severity == severity])
            summary['alerts_by_severity'][severity.value] = count
        
        # Count by type
        for alert_type in AlertType:
            count = len([alert for alert in self.active_alerts if alert.alert_type == alert_type])
            summary['alerts_by_type'][alert_type.value] = count
        
        # Most affected keywords
        keyword_mentions = {}
        for alert in self.active_alerts:
            for keyword in alert.affected_keywords:
                keyword_mentions[keyword] = keyword_mentions.get(keyword, 0) + 1
        
        summary['most_affected_keywords'] = sorted(
            keyword_mentions.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        return summary
    
    def _determine_monitoring_status(self) -> str:
        """Determine overall monitoring status"""
        
        if not self.active_alerts:
            return 'healthy'
        
        critical_count = len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL])
        high_count = len([a for a in self.active_alerts if a.severity == AlertSeverity.HIGH])
        
        if critical_count > 0:
            return 'critical'
        elif high_count >= 3:
            return 'warning'
        elif len(self.active_alerts) > 10:
            return 'attention_needed'
        else:
            return 'stable'
    
    def get_alert_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive alert dashboard data"""
        
        return {
            'current_status': self._determine_monitoring_status(),
            'active_alerts': [self._serialize_alert(alert) for alert in self.active_alerts],
            'alert_summary': self._generate_alert_summary(),
            'recent_history': [
                self._serialize_alert(alert) for alert in self.alert_history[-10:]
            ],
            'monitoring_metrics': {
                'total_alerts_today': len([
                    alert for alert in self.active_alerts 
                    if alert.timestamp.date() == datetime.now().date()
                ]),
                'avg_resolution_time': self._calculate_avg_resolution_time(),
                'alert_frequency': self._calculate_alert_frequency()
            }
        }
    
    def _serialize_alert(self, alert: Alert) -> Dict[str, Any]:
        """Serialize alert for JSON output"""
        
        return {
            'alert_id': alert.alert_id,
            'alert_type': alert.alert_type.value,
            'severity': alert.severity.value,
            'title': alert.title,
            'description': alert.description,
            'affected_keywords_count': len(alert.affected_keywords),
            'top_affected_keywords': alert.affected_keywords[:5],
            'metrics': alert.metrics,
            'timestamp': alert.timestamp.isoformat(),
            'recommended_actions': alert.recommended_actions,
            'auto_resolve': alert.auto_resolve
        }
    
    def _calculate_avg_resolution_time(self) -> float:
        """Calculate average alert resolution time"""
        """Calculate average alert resolution time from actual alert history"""
        
        if not self.alert_history:
            return 0.0
        
        # Calculate resolution times for resolved alerts
        resolution_times = []
        current_time = datetime.now()
        
        for alert in self.alert_history:
            # For resolved alerts, calculate how long they were active
            # This assumes alerts in history were resolved when moved there
            alert_age_hours = (current_time - alert.timestamp).total_seconds() / 3600
            
            # Convert to days and add to resolution times
            resolution_days = alert_age_hours / 24
            resolution_times.append(resolution_days)
        
        if resolution_times:
            return sum(resolution_times) / len(resolution_times)
        
        return 0.0
    
    def _calculate_alert_frequency(self) -> Dict[str, float]:
        """Calculate alert frequency metrics"""
        """Calculate alert frequency metrics from actual data"""
        
        now = datetime.now()
        all_alerts = self.active_alerts + self.alert_history
        
        # Count alerts by time period
        today_alerts = len([
            alert for alert in all_alerts
            if alert.timestamp.date() == now.date()
        ])
        
        yesterday = now - timedelta(days=1)
        yesterday_alerts = len([
            alert for alert in all_alerts
            if alert.timestamp.date() == yesterday.date()
        ])
        
        week_alerts = len([
            alert for alert in all_alerts
            if (now - alert.timestamp).days <= 7
        ])
        
        month_alerts = len([
            alert for alert in all_alerts
            if (now - alert.timestamp).days <= 30
        ])
        
        # Calculate trends
        daily_trend = 'stable'
        if yesterday_alerts > 0:
            daily_change = (today_alerts - yesterday_alerts) / yesterday_alerts
            if daily_change > 0.2:
                daily_trend = 'increasing'
            elif daily_change < -0.2:
                daily_trend = 'decreasing'
        
        return {
            'alerts_today': today_alerts,
            'alerts_yesterday': yesterday_alerts,
            'alerts_this_week': week_alerts,
            'alerts_this_month': month_alerts,
            'daily_average': week_alerts / 7 if week_alerts > 0 else 0,
            'weekly_average': month_alerts / 4 if month_alerts > 0 else 0,
            'daily_trend': daily_trend,
            'total_alerts_tracked': len(all_alerts)
        }

    def get_traffic_drop_keywords(self, domain: str, date: str, threshold: float = 1.0) -> Dict[str, Any]:
        """Get keywords with >1% traffic drop since date using real data"""
        
        # Find the brand data for the domain
        brand_key = None
        for brand in ['lenovo', 'dell', 'hp']: # Assuming these are the brands monitored
            if domain.lower() in brand or brand in domain.lower():
                brand_key = brand
                break
        
        if not brand_key:
            return {
                'dropping_keywords': [],
                'total_impact': 0,
                'recovery_timeline': {},
                'immediate_actions': []
            }
        
        # Get current and historical data for comparison
        current_alerts = [alert for alert in self.active_alerts 
                         if alert.alert_type == AlertType.TRAFFIC_DROP]
        
        dropping_keywords = []
        total_impact = 0
        
        for alert in current_alerts:
            if brand_key in alert.alert_id.lower(): # Match alert to the brand
                # Extract keywords from alert metrics
                affected_keywords = alert.affected_keywords
                avg_drop = alert.metrics.get('avg_drop_percent', 0)
                
                if abs(avg_drop) >= threshold:
                    for keyword in affected_keywords:
                        dropping_keywords.append({
                            'keyword': keyword,
                            'traffic_drop_percent': avg_drop,
                            'alert_severity': alert.severity.value,
                            'detected_date': alert.timestamp.strftime('%Y-%m-%d'),
                            'recovery_priority': 'high' if abs(avg_drop) > 50 else 'medium'
                        })
                    
                    total_impact += abs(avg_drop) # Summing average drops might not be accurate, consider total_traffic_lost
        
        # Sort by drop percentage
        dropping_keywords.sort(key=lambda x: abs(x['traffic_drop_percent']), reverse=True)
        
        # Categorize by recovery timeline
        recovery_timeline = {
            'immediate': [kw for kw in dropping_keywords[:20] if abs(kw['traffic_drop_percent']) > 50],
            'short_term': [kw for kw in dropping_keywords if 20 <= abs(kw['traffic_drop_percent']) <= 50],
            'long_term': [kw for kw in dropping_keywords if abs(kw['traffic_drop_percent']) < 20]
        }
        
        # Generate immediate actions based on actual drops detected
        immediate_actions = []
        if len(recovery_timeline['immediate']) > 0:
            immediate_actions.append(f"Emergency review of {len(recovery_timeline['immediate'])} critical keywords")
        if total_impact > 100: # Example threshold for comprehensive audit
            immediate_actions.append("Comprehensive traffic recovery audit required")
        
        immediate_actions.extend([
            "Technical SEO health check",
            "Content quality assessment for affected keywords",
            "Competitor movement analysis",
            "Algorithm update correlation check"
        ])
        
        return {
            'dropping_keywords': dropping_keywords[:20],  # Top 20 affected
            'total_impact': total_impact,
            'recovery_timeline': recovery_timeline,
            'immediate_actions': immediate_actions,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_keywords_analyzed': len(dropping_keywords)
        }

    def analyze_historical_alert_patterns(self) -> Dict[str, Any]:
        """Analyze patterns from actual alert history"""
        
        if not self.alert_history:
            return {'status': 'no_historical_data'}
        
        patterns = {
            'most_common_alert_types': {},
            'peak_alert_times': {},
            'average_severity_distribution': {},
            'keyword_vulnerability_patterns': {}, # Placeholder for future enhancement
            'resolution_success_rate': 0.0 # Assuming alerts in history are "resolved"
        }
        
        # Analyze alert type frequency
        type_counts = {}
        for alert in self.alert_history:
            alert_type = alert.alert_type.value
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        patterns['most_common_alert_types'] = sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True
        )
        
        # Analyze timing patterns (e.g., hour of day)
        hour_counts = {}
        for alert in self.alert_history:
            hour = alert.timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        if hour_counts:
            peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0]
            patterns['peak_alert_times'] = {
                'peak_hour': peak_hour,
                'hourly_distribution': hour_counts
            }
        
        # Analyze severity distribution
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        patterns['average_severity_distribution'] = severity_counts
        
        # Calculate resolution success rate (simplified: resolved / total)
        total_alerts_ever = len(self.active_alerts) + len(self.alert_history)
        if total_alerts_ever > 0:
            patterns['resolution_success_rate'] = len(self.alert_history) / total_alerts_ever
        
        return patterns
