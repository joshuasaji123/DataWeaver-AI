"""
Visualization Validator

This module provides intelligent validation for data visualization,
ensuring agents only create charts when data is suitable and preventing empty graphs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class VisualizationValidator:
    """
    Intelligent validator for data visualization suitability
    """
    
    def __init__(self):
        """Initialize the visualization validator"""
        self.min_data_points = 2
        self.min_unique_values = 2
        self.max_categories_for_pie = 10
        self.max_categories_for_bar = 20
        self.min_numeric_columns = 1
        
        # Chart type requirements
        self.chart_requirements = {
            'histogram': {
                'min_numeric_cols': 1,
                'min_data_points': 5,
                'data_types': ['numeric'],
                'description': 'Distribution of numeric values'
            },
            'scatter': {
                'min_numeric_cols': 2,
                'min_data_points': 3,
                'data_types': ['numeric'],
                'description': 'Relationship between two numeric variables'
            },
            'line': {
                'min_numeric_cols': 1,
                'min_data_points': 2,
                'data_types': ['numeric', 'datetime'],
                'description': 'Trends over time or sequential data'
            },
            'bar': {
                'min_categorical_cols': 1,
                'min_data_points': 1,
                'max_categories': 20,
                'data_types': ['categorical', 'numeric'],
                'description': 'Comparison of categories'
            },
            'pie': {
                'min_categorical_cols': 1,
                'min_data_points': 2,
                'max_categories': 10,
                'data_types': ['categorical'],
                'description': 'Proportion of categories'
            },
            'box': {
                'min_numeric_cols': 1,
                'min_data_points': 5,
                'data_types': ['numeric'],
                'description': 'Distribution and outliers in numeric data'
            },
            'heatmap': {
                'min_numeric_cols': 2,
                'min_data_points': 4,
                'data_types': ['numeric'],
                'description': 'Correlation between multiple variables'
            },
            'time_series': {
                'min_datetime_cols': 1,
                'min_numeric_cols': 1,
                'min_data_points': 3,
                'data_types': ['datetime', 'numeric'],
                'description': 'Changes over time'
            }
        }
    
    def validate_data_for_visualization(self, data: pd.DataFrame) -> Dict:
        """
        Validate if data is suitable for visualization
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        if data is None or data.empty:
            return {
                'is_suitable': False,
                'reason': 'Data is empty or None',
                'recommendations': ['Upload valid data before attempting visualization'],
                'suitable_charts': [],
                'data_summary': {}
            }
        
        # Analyze data characteristics
        data_analysis = self._analyze_data_characteristics(data)
        
        # Check basic requirements
        basic_validation = self._check_basic_requirements(data, data_analysis)
        
        if not basic_validation['is_suitable']:
            return basic_validation
        
        # Determine suitable chart types
        suitable_charts = self._determine_suitable_charts(data, data_analysis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(data, data_analysis, suitable_charts)
        
        return {
            'is_suitable': len(suitable_charts) > 0,
            'reason': 'Data is suitable for visualization' if suitable_charts else 'No suitable chart types found',
            'suitable_charts': suitable_charts,
            'recommendations': recommendations,
            'data_summary': data_analysis,
            'chart_suggestions': self._get_chart_suggestions(suitable_charts)
        }
    
    def _analyze_data_characteristics(self, data: pd.DataFrame) -> Dict:
        """Analyze data characteristics for visualization suitability"""
        analysis = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'boolean_columns': [],
            'null_percentage': {},
            'unique_value_counts': {},
            'data_types': {}
        }
        
        for col in data.columns:
            col_data = data[col]
            
            # Determine data type
            if pd.api.types.is_numeric_dtype(col_data):
                analysis['numeric_columns'].append(col)
                analysis['data_types'][col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                analysis['datetime_columns'].append(col)
                analysis['data_types'][col] = 'datetime'
            elif pd.api.types.is_bool_dtype(col_data):
                analysis['boolean_columns'].append(col)
                analysis['data_types'][col] = 'boolean'
            else:
                analysis['categorical_columns'].append(col)
                analysis['data_types'][col] = 'categorical'
            
            # Calculate null percentage
            null_pct = (col_data.isnull().sum() / len(col_data)) * 100
            analysis['null_percentage'][col] = round(null_pct, 2)
            
            # Count unique values
            analysis['unique_value_counts'][col] = col_data.nunique()
        
        return analysis
    
    def _check_basic_requirements(self, data: pd.DataFrame, analysis: Dict) -> Dict:
        """Check basic requirements for any visualization"""
        
        # Check minimum data points
        if analysis['row_count'] < self.min_data_points:
            return {
                'is_suitable': False,
                'reason': f'Insufficient data points (need at least {self.min_data_points}, have {analysis["row_count"]})',
                'recommendations': ['Collect more data before attempting visualization'],
                'suitable_charts': [],
                'data_summary': analysis
            }
        
        # Check for at least one non-null column
        valid_columns = []
        for col in data.columns:
            if analysis['null_percentage'][col] < 100:  # Not entirely null
                valid_columns.append(col)
        
        if not valid_columns:
            return {
                'is_suitable': False,
                'reason': 'All columns contain only null values',
                'recommendations': ['Clean the data to remove null values', 'Check data quality'],
                'suitable_charts': [],
                'data_summary': analysis
            }
        
        # Check for at least one column with variation
        has_variation = False
        for col in valid_columns:
            if analysis['unique_value_counts'][col] >= self.min_unique_values:
                has_variation = True
                break
        
        if not has_variation:
            return {
                'is_suitable': False,
                'reason': 'No columns have sufficient variation (all values are the same)',
                'recommendations': ['Check data quality', 'Ensure data contains varied values'],
                'suitable_charts': [],
                'data_summary': analysis
            }
        
        return {
            'is_suitable': True,
            'reason': 'Basic requirements met',
            'data_summary': analysis
        }
    
    def _determine_suitable_charts(self, data: pd.DataFrame, analysis: Dict) -> List[Dict]:
        """Determine which chart types are suitable for the data"""
        suitable_charts = []
        
        for chart_type, requirements in self.chart_requirements.items():
            is_suitable, reason = self._check_chart_suitability(data, analysis, chart_type, requirements)
            
            if is_suitable:
                suitable_charts.append({
                    'type': chart_type,
                    'description': requirements['description'],
                    'reason': reason,
                    'priority': self._calculate_chart_priority(chart_type, analysis)
                })
        
        # Sort by priority (higher is better)
        suitable_charts.sort(key=lambda x: x['priority'], reverse=True)
        
        return suitable_charts
    
    def _check_chart_suitability(self, data: pd.DataFrame, analysis: Dict, 
                                chart_type: str, requirements: Dict) -> Tuple[bool, str]:
        """Check if a specific chart type is suitable for the data"""
        
        # Check numeric column requirements
        if 'min_numeric_cols' in requirements:
            if len(analysis['numeric_columns']) < requirements['min_numeric_cols']:
                return False, f"Need {requirements['min_numeric_cols']} numeric columns, have {len(analysis['numeric_columns'])}"
        
        # Check categorical column requirements
        if 'min_categorical_cols' in requirements:
            if len(analysis['categorical_columns']) < requirements['min_categorical_cols']:
                return False, f"Need {requirements['min_categorical_cols']} categorical columns, have {len(analysis['categorical_columns'])}"
        
        # Check datetime column requirements
        if 'min_datetime_cols' in requirements:
            if len(analysis['datetime_columns']) < requirements['min_datetime_cols']:
                return False, f"Need {requirements['min_datetime_cols']} datetime columns, have {len(analysis['datetime_columns'])}"
        
        # Check data point requirements
        if 'min_data_points' in requirements:
            if analysis['row_count'] < requirements['min_data_points']:
                return False, f"Need {requirements['min_data_points']} data points, have {analysis['row_count']}"
        
        # Check category limits for categorical charts
        if chart_type in ['pie', 'bar'] and analysis['categorical_columns']:
            max_categories = requirements.get('max_categories', float('inf'))
            for col in analysis['categorical_columns']:
                if analysis['unique_value_counts'][col] > max_categories:
                    if chart_type == 'pie':
                        return False, f"Too many categories in '{col}' for pie chart ({analysis['unique_value_counts'][col]} > {max_categories})"
                    # For bar charts, we can suggest grouping instead of rejection
        
        # Special checks for specific chart types
        if chart_type == 'scatter':
            # Need at least 2 numeric columns with sufficient variation
            numeric_cols_with_variation = [
                col for col in analysis['numeric_columns'] 
                if analysis['unique_value_counts'][col] >= 3
            ]
            if len(numeric_cols_with_variation) < 2:
                return False, "Need 2 numeric columns with variation for scatter plot"
        
        elif chart_type == 'heatmap':
            # Need multiple numeric columns for correlation
            if len(analysis['numeric_columns']) < 2:
                return False, "Need multiple numeric columns for correlation heatmap"
        
        elif chart_type == 'time_series':
            # Need datetime and numeric columns
            if not analysis['datetime_columns'] or not analysis['numeric_columns']:
                return False, "Need both datetime and numeric columns for time series"
        
        return True, f"Suitable for {chart_type}: {requirements['description']}"
    
    def _calculate_chart_priority(self, chart_type: str, analysis: Dict) -> int:
        """Calculate priority score for chart types based on data characteristics"""
        priority = 0
        
        # Base priorities
        base_priorities = {
            'histogram': 7,
            'bar': 8,
            'scatter': 6,
            'line': 7,
            'pie': 5,
            'box': 6,
            'heatmap': 4,
            'time_series': 9  # High priority if datetime data is present
        }
        
        priority += base_priorities.get(chart_type, 5)
        
        # Boost priority based on data characteristics
        if chart_type == 'time_series' and analysis['datetime_columns']:
            priority += 3
        
        if chart_type == 'bar' and analysis['categorical_columns']:
            priority += 2
        
        if chart_type == 'histogram' and analysis['numeric_columns']:
            priority += 1
        
        # Reduce priority for charts with too many categories
        if chart_type in ['pie', 'bar'] and analysis['categorical_columns']:
            max_categories = max([analysis['unique_value_counts'][col] for col in analysis['categorical_columns']], default=0)
            if max_categories > 15:
                priority -= 2
        
        return priority
    
    def _generate_recommendations(self, data: pd.DataFrame, analysis: Dict, 
                                suitable_charts: List[Dict]) -> List[str]:
        """Generate recommendations for data visualization"""
        recommendations = []
        
        if not suitable_charts:
            recommendations.append("No suitable visualizations found for current data")
            recommendations.append("Consider data preprocessing or transformation")
            return recommendations
        
        # Recommend top chart types
        top_charts = suitable_charts[:3]
        chart_names = [chart['type'] for chart in top_charts]
        recommendations.append(f"Recommended chart types: {', '.join(chart_names)}")
        
        # Data quality recommendations
        high_null_cols = [col for col, pct in analysis['null_percentage'].items() if pct > 20]
        if high_null_cols:
            recommendations.append(f"Consider handling missing values in: {', '.join(high_null_cols)}")
        
        # Category count recommendations
        if analysis['categorical_columns']:
            high_category_cols = [
                col for col in analysis['categorical_columns'] 
                if analysis['unique_value_counts'][col] > 15
            ]
            if high_category_cols:
                recommendations.append(f"Consider grouping categories in: {', '.join(high_category_cols)}")
        
        # Data size recommendations
        if analysis['row_count'] > 10000:
            recommendations.append("Consider sampling for better visualization performance")
        elif analysis['row_count'] < 10:
            recommendations.append("More data points would improve visualization quality")
        
        return recommendations
    
    def _get_chart_suggestions(self, suitable_charts: List[Dict]) -> List[Dict]:
        """Get specific chart suggestions with columns to use"""
        suggestions = []
        
        for chart in suitable_charts[:5]:  # Top 5 suggestions
            suggestion = {
                'chart_type': chart['type'],
                'description': chart['description'],
                'priority': chart['priority']
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def validate_specific_chart(self, data: pd.DataFrame, chart_type: str, 
                              columns: List[str] = None) -> Dict:
        """
        Validate if specific chart type and columns are suitable
        
        Args:
            data: DataFrame to validate
            chart_type: Type of chart to validate
            columns: Specific columns to use (optional)
            
        Returns:
            Validation result for specific chart
        """
        if data is None or data.empty:
            return {
                'is_valid': False,
                'reason': 'Data is empty',
                'suggestions': []
            }
        
        # If specific columns are provided, validate them
        if columns:
            missing_cols = [col for col in columns if col not in data.columns]
            if missing_cols:
                return {
                    'is_valid': False,
                    'reason': f'Columns not found: {missing_cols}',
                    'suggestions': [f'Available columns: {list(data.columns)}']
                }
            
            # Use only specified columns for validation
            subset_data = data[columns]
        else:
            subset_data = data
        
        # Analyze subset data
        analysis = self._analyze_data_characteristics(subset_data)
        
        # Check if chart type is supported
        if chart_type not in self.chart_requirements:
            return {
                'is_valid': False,
                'reason': f'Chart type "{chart_type}" not supported',
                'suggestions': [f'Supported types: {list(self.chart_requirements.keys())}']
            }
        
        # Validate chart requirements
        requirements = self.chart_requirements[chart_type]
        is_suitable, reason = self._check_chart_suitability(subset_data, analysis, chart_type, requirements)
        
        suggestions = []
        if not is_suitable:
            suggestions = self._get_improvement_suggestions(chart_type, analysis, requirements)
        
        return {
            'is_valid': is_suitable,
            'reason': reason,
            'suggestions': suggestions,
            'data_analysis': analysis
        }
    
    def _get_improvement_suggestions(self, chart_type: str, analysis: Dict, 
                                   requirements: Dict) -> List[str]:
        """Get suggestions for improving data for specific chart type"""
        suggestions = []
        
        # Suggest data transformations
        if 'min_numeric_cols' in requirements and len(analysis['numeric_columns']) < requirements['min_numeric_cols']:
            suggestions.append(f"Convert categorical columns to numeric if possible")
            suggestions.append(f"Consider alternative chart types that work with categorical data")
        
        if 'min_categorical_cols' in requirements and len(analysis['categorical_columns']) < requirements['min_categorical_cols']:
            suggestions.append("Group numeric data into categories")
            suggestions.append("Consider binning continuous variables")
        
        if analysis['row_count'] < requirements.get('min_data_points', 0):
            suggestions.append("Collect more data points")
            suggestions.append("Consider simpler chart types that require less data")
        
        return suggestions


# Global instance for easy access
visualization_validator = VisualizationValidator() 