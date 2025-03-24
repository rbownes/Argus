"""
Dashboard logic and data aggregation for visualization service.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import logging
from visualization_service.database import VisualizationDB

logger = logging.getLogger("visualization_service.dashboard")

class Dashboard:
    """Dashboard data processing and aggregation."""
    
    def __init__(self, db: VisualizationDB):
        """
        Initialize dashboard.
        
        Args:
            db: Database instance
        """
        self.db = db
        self.logger = logging.getLogger("visualization_service.dashboard")
    
    async def get_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        themes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get dashboard summary.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            models: List of model IDs to include
            themes: List of themes to include
            
        Returns:
            Dictionary with summary statistics
        """
        self.logger.info("Generating dashboard summary")
        try:
            summary = self.db.get_dashboard_summary(start_date, end_date, models, themes)
            
            # Add available filter options
            summary["available_models"] = self.db.get_available_models()
            summary["available_themes"] = self.db.get_available_themes()
            
            return summary
        except Exception as e:
            self.logger.error(f"Error generating dashboard summary: {str(e)}")
            return {"error": str(e)}
    
    async def get_model_performance_timeline(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
        time_grouping: str = "day"
    ) -> Dict[str, Any]:
        """
        Get model performance timeline data.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            models: List of model IDs to include
            themes: List of themes to include
            time_grouping: Time grouping (day, week, month)
            
        Returns:
            Dictionary with timeline data
        """
        self.logger.info(f"Generating model performance timeline with {time_grouping} grouping")
        try:
            # Validate time_grouping
            valid_groupings = ["day", "week", "month"]
            if time_grouping not in valid_groupings:
                time_grouping = "day"
            
            # Get data from database
            df = self.db.get_model_performance_over_time(
                start_date, end_date, models, themes, time_grouping
            )
            
            if df.empty:
                return {"series": [], "labels": []}
            
            # Format for Chart.js
            result = {"series": [], "labels": []}
            
            # Format time periods for x-axis labels
            time_periods = sorted(df['time_period'].unique())
            result["labels"] = [t.strftime("%Y-%m-%d") for t in time_periods]
            
            # Create series for each model
            for model_id in df['model_id'].unique():
                model_data = df[df['model_id'] == model_id]
                
                # If themes are specified, separate by theme
                if themes and len(themes) > 1:
                    for theme in model_data['theme'].unique():
                        theme_data = model_data[model_data['theme'] == theme]
                        
                        # Create score data aligned with time_periods
                        scores = []
                        for period in time_periods:
                            period_data = theme_data[theme_data['time_period'] == period]
                            if not period_data.empty:
                                scores.append(float(period_data['avg_score'].values[0]))
                            else:
                                scores.append(None)  # Use None for missing data points
                        
                        result["series"].append({
                            "name": f"{model_id} - {theme}",
                            "data": scores
                        })
                else:
                    # Aggregate across themes
                    period_scores = {}
                    for _, row in model_data.iterrows():
                        period = row['time_period']
                        if period not in period_scores:
                            period_scores[period] = {"sum": 0, "count": 0}
                        
                        period_scores[period]["sum"] += row['avg_score'] * row['eval_count']
                        period_scores[period]["count"] += row['eval_count']
                    
                    # Create score data aligned with time_periods
                    scores = []
                    for period in time_periods:
                        if period in period_scores and period_scores[period]["count"] > 0:
                            avg = period_scores[period]["sum"] / period_scores[period]["count"]
                            scores.append(float(avg))
                        else:
                            scores.append(None)  # Use None for missing data points
                    
                    result["series"].append({
                        "name": model_id,
                        "data": scores
                    })
            
            return result
        except Exception as e:
            self.logger.error(f"Error generating model performance timeline: {str(e)}")
            return {"error": str(e), "series": [], "labels": []}
    
    async def get_model_comparison(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        themes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get model comparison data.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            models: List of model IDs to include
            themes: List of themes to include
            
        Returns:
            Dictionary with model comparison data
        """
        self.logger.info("Generating model comparison data")
        try:
            # Get data from database
            df = self.db.get_model_comparison(start_date, end_date, models, themes)
            
            if df.empty:
                return {"models": [], "themes": [], "data": []}
            
            # Format for charts
            unique_models = sorted(df['model_id'].unique())
            unique_themes = sorted(df['theme'].unique())
            
            # Bar chart data (average scores by model)
            bar_data = {
                "labels": unique_models,
                "datasets": []
            }
            
            if len(unique_themes) > 1:
                # Multiple themes - create a dataset for each theme
                for theme in unique_themes:
                    theme_data = df[df['theme'] == theme]
                    
                    # Get scores for each model
                    scores = []
                    for model in unique_models:
                        model_data = theme_data[theme_data['model_id'] == model]
                        if not model_data.empty:
                            scores.append(float(model_data['avg_score'].values[0]))
                        else:
                            scores.append(0)  # Use 0 for missing data
                    
                    bar_data["datasets"].append({
                        "label": theme,
                        "data": scores
                    })
            else:
                # Single theme or aggregated - one dataset with average by model
                scores = []
                for model in unique_models:
                    model_data = df[df['model_id'] == model]
                    if not model_data.empty:
                        # Average across themes if multiple exist
                        avg_score = (model_data['avg_score'] * model_data['eval_count']).sum() / model_data['eval_count'].sum()
                        scores.append(float(avg_score))
                    else:
                        scores.append(0)  # Use 0 for missing data
                
                bar_data["datasets"].append({
                    "label": "Average Score",
                    "data": scores
                })
            
            # Radar chart data (performance across themes)
            radar_data = {
                "labels": unique_themes,
                "datasets": []
            }
            
            for model in unique_models:
                model_data = df[df['model_id'] == model]
                
                # Get scores for each theme
                scores = []
                for theme in unique_themes:
                    theme_data = model_data[model_data['theme'] == theme]
                    if not theme_data.empty:
                        scores.append(float(theme_data['avg_score'].values[0]))
                    else:
                        scores.append(0)  # Use 0 for missing data
                
                radar_data["datasets"].append({
                    "label": model,
                    "data": scores
                })
            
            # Score distribution data
            histogram_data = self.db.get_score_distribution(
                start_date, end_date, models, themes
            )
            
            return {
                "bar_chart": bar_data,
                "radar_chart": radar_data,
                "histograms": histogram_data,
                "raw_data": df.to_dict(orient="records")
            }
        except Exception as e:
            self.logger.error(f"Error generating model comparison: {str(e)}")
            return {"error": str(e)}
    
    async def get_theme_analysis(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        themes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get theme analysis data.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            models: List of model IDs to include
            themes: List of themes to include
            
        Returns:
            Dictionary with theme analysis data
        """
        self.logger.info("Generating theme analysis data")
        try:
            # Get heatmap data
            heatmap_df = self.db.get_theme_analysis(start_date, end_date, models, themes)
            
            if heatmap_df.empty:
                return {"heatmap": {"x": [], "y": [], "z": []}}
            
            # Format for heatmap (plotly)
            x_labels = heatmap_df.columns.tolist()  # themes
            y_labels = heatmap_df.index.tolist()    # models
            
            # Create z values (2D array)
            z_values = []
            for model in y_labels:
                row = []
                for theme in x_labels:
                    row.append(float(heatmap_df.loc[model, theme]) if not pd.isna(heatmap_df.loc[model, theme]) else 0)
                z_values.append(row)
            
            heatmap_data = {
                "x": x_labels,
                "y": y_labels,
                "z": z_values
            }
            
            # Theme performance over time
            timeline_df = self.db.get_model_performance_over_time(
                start_date, end_date, models, themes, "week"
            )
            
            theme_timeline = {"labels": [], "datasets": []}
            
            if not timeline_df.empty:
                time_periods = sorted(timeline_df['time_period'].unique())
                theme_timeline["labels"] = [t.strftime("%Y-%m-%d") for t in time_periods]
                
                # Create a dataset for each theme
                for theme in timeline_df['theme'].unique():
                    theme_data = timeline_df[timeline_df['theme'] == theme]
                    
                    # Aggregate across models for each time period
                    theme_scores = []
                    for period in time_periods:
                        period_data = theme_data[theme_data['time_period'] == period]
                        if not period_data.empty:
                            # Weighted average by eval_count
                            avg_score = (period_data['avg_score'] * period_data['eval_count']).sum() / period_data['eval_count'].sum()
                            theme_scores.append(float(avg_score))
                        else:
                            theme_scores.append(None)  # Use None for missing data points
                    
                    theme_timeline["datasets"].append({
                        "label": theme,
                        "data": theme_scores
                    })
            
            return {
                "heatmap": heatmap_data,
                "theme_timeline": theme_timeline,
                "raw_data": heatmap_df.reset_index().melt(
                    id_vars=['model_id'], 
                    var_name='theme', 
                    value_name='score'
                ).dropna().to_dict(orient="records")
            }
        except Exception as e:
            self.logger.error(f"Error generating theme analysis: {str(e)}")
            return {"error": str(e)}
    
    async def get_detailed_results(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
        evaluation_prompt_ids: Optional[List[str]] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        page: int = 1,
        page_size: int = 10,
        sort_by: str = "timestamp",
        sort_desc: bool = True
    ) -> Dict[str, Any]:
        """
        Get detailed evaluation results with pagination.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            models: List of model IDs to include
            themes: List of themes to include
            evaluation_prompt_ids: List of evaluation prompt IDs to include
            min_score: Minimum score
            max_score: Maximum score
            page: Page number (1-based)
            page_size: Number of results per page
            sort_by: Column to sort by
            sort_desc: Whether to sort in descending order
            
        Returns:
            Dictionary with paginated results and metadata
        """
        self.logger.info(f"Getting detailed results (page {page}, size {page_size})")
        try:
            # Calculate offset
            offset = (page - 1) * page_size if page > 0 else 0
            
            # Get results with pagination
            results, total_count = self.db.get_filtered_results(
                start_date=start_date,
                end_date=end_date,
                models=models,
                themes=themes,
                evaluation_prompt_ids=evaluation_prompt_ids,
                min_score=min_score,
                max_score=max_score,
                limit=page_size,
                offset=offset,
                sort_by=sort_by,
                sort_desc=sort_desc
            )
            
            # Calculate pagination metadata
            total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 0
            
            return {
                "results": results,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1
                },
                "filters": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "models": models,
                    "themes": themes,
                    "evaluation_prompt_ids": evaluation_prompt_ids,
                    "min_score": min_score,
                    "max_score": max_score,
                    "sort_by": sort_by,
                    "sort_desc": sort_desc
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting detailed results: {str(e)}")
            return {"error": str(e), "results": [], "pagination": {"total_count": 0}}
    
    async def get_filter_options(self) -> Dict[str, List]:
        """
        Get available filter options for the dashboard.
        
        Returns:
            Dictionary with available models, themes, and evaluation prompts
        """
        self.logger.info("Getting filter options")
        try:
            return {
                "models": self.db.get_available_models(),
                "themes": self.db.get_available_themes(),
                "evaluation_prompts": self.db.get_available_evaluation_prompts()
            }
        except Exception as e:
            self.logger.error(f"Error getting filter options: {str(e)}")
            return {"models": [], "themes": [], "evaluation_prompts": []}
