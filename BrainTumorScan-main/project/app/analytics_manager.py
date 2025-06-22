import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from csv_manager import CSVManager
from model_manager import BrainTumorModelManager
import numpy as np

class AnalyticsManager:
    def __init__(self, csv_manager: CSVManager, model_manager: BrainTumorModelManager):
        self.csv_manager = csv_manager
        self.model_manager = model_manager
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        try:
            # Get data from CSV files
            predictions = self.csv_manager.read_csv('predictions.csv')
            dataset = self.csv_manager.read_csv('dataset.csv')
            training_sessions = self.csv_manager.read_csv('training_sessions.csv')
            
            # Calculate basic stats
            total_scans = len(predictions)
            detected_tumors = len([p for p in predictions if p['has_tumor'].lower() == 'true'])
            
            # Get current model accuracy
            current_model_accuracy = 0.0
            if self.model_manager.current_model_name:
                try:
                    current_model = self.model_manager.current_model_name
                    if current_model in self.model_manager.models:
                        version_manager = self.model_manager.models[current_model]
                        if version_manager.current_version:
                            current_version = version_manager.versions[version_manager.current_version]
                            current_model_accuracy = current_version.metadata.get('test_accuracy', 0.0)
                except:
                    pass
            
            # Calculate dataset stats
            dataset_size = len(dataset)
            dataset_tumor_count = len([d for d in dataset if d['label'] == 'tumor'])
            
            # Calculate training stats
            completed_trainings = len([t for t in training_sessions if t['status'] == 'complete'])
            
            # Calculate detection rate
            detection_rate = round((detected_tumors / total_scans * 100) if total_scans > 0 else 0, 1)
            
            return {
                'total_scans': total_scans,
                'detected_tumors': detected_tumors,
                'detection_rate': detection_rate,
                'model_accuracy': round(current_model_accuracy * 100, 1) if current_model_accuracy else 0.0,
                'dataset_size': dataset_size,
                'dataset_tumor_ratio': round((dataset_tumor_count / dataset_size * 100) if dataset_size > 0 else 0, 1),
                'completed_trainings': completed_trainings,
                'current_model': self.model_manager.current_model_name or 'None',
                'system_status': 'online'
            }
            
        except Exception as e:
            return {
                'error': f'Failed to get system stats: {str(e)}',
                'system_status': 'error'
            }
    
    def get_usage_analytics(self, period: str = 'week') -> Dict[str, Any]:
        """Get usage analytics over time"""
        try:
            predictions = self.csv_manager.read_csv('predictions.csv')
            
            # Parse timestamps and sort by date
            prediction_data = []
            for pred in predictions:
                try:
                    timestamp = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                    prediction_data.append({
                        'timestamp': timestamp,
                        'has_tumor': pred['has_tumor'].lower() == 'true',
                        'confidence': float(pred['confidence']),
                        'model_used': pred['model_used']
                    })
                except:
                    continue
            
            # Generate time series data based on period
            if period == 'day':
                days_back = 7
                time_format = '%H:00'
                group_by = 'hour'
            elif period == 'week':
                days_back = 30
                time_format = '%m/%d'
                group_by = 'day'
            elif period == 'month':
                days_back = 365
                time_format = '%b'
                group_by = 'month'
            else:
                days_back = 30
                time_format = '%m/%d'
                group_by = 'day'
            
            # Generate time series
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            time_series = self._generate_time_series(
                prediction_data, start_date, end_date, group_by
            )
            
            # Calculate trends
            total_predictions_trend = [point['total'] for point in time_series]
            tumor_detections_trend = [point['tumor_count'] for point in time_series]
            confidence_trend = [point['avg_confidence'] for point in time_series]
            
            return {
                'period': period,
                'time_series': time_series,
                'trends': {
                    'total_predictions': total_predictions_trend,
                    'tumor_detections': tumor_detections_trend,
                    'average_confidence': confidence_trend
                },
                'summary': {
                    'total_period_predictions': sum(total_predictions_trend),
                    'total_period_tumors': sum(tumor_detections_trend),
                    'average_confidence': round(np.mean([c for c in confidence_trend if c > 0]), 3) if confidence_trend else 0.0
                }
            }
            
        except Exception as e:
            return {
                'error': f'Failed to get usage analytics: {str(e)}',
                'period': period,
                'time_series': [],
                'trends': {},
                'summary': {}
            }
    
    def _generate_time_series(self, prediction_data: List[Dict], start_date: datetime, 
                            end_date: datetime, group_by: str) -> List[Dict[str, Any]]:
        """Generate time series data grouped by specified period"""
        time_series = []
        
        if group_by == 'hour':
            current = start_date.replace(minute=0, second=0, microsecond=0)
            delta = timedelta(hours=1)
        elif group_by == 'day':
            current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            delta = timedelta(days=1)
        elif group_by == 'month':
            current = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            delta = timedelta(days=30)  # Approximate
        else:
            current = start_date
            delta = timedelta(days=1)
        
        while current <= end_date:
            next_period = current + delta
            
            # Filter predictions for this time period
            period_predictions = [
                p for p in prediction_data 
                if current <= p['timestamp'] < next_period
            ]
            
            # Calculate metrics for this period
            total_count = len(period_predictions)
            tumor_count = len([p for p in period_predictions if p['has_tumor']])
            
            confidences = [p['confidence'] for p in period_predictions]
            avg_confidence = round(np.mean(confidences), 3) if confidences else 0.0
            
            # Format label based on grouping
            if group_by == 'hour':
                label = current.strftime('%H:00')
            elif group_by == 'day':
                label = current.strftime('%m/%d')
            elif group_by == 'month':
                label = current.strftime('%b')
            else:
                label = current.strftime('%m/%d')
            
            time_series.append({
                'label': label,
                'timestamp': current.isoformat(),
                'total': total_count,
                'tumor_count': tumor_count,
                'no_tumor_count': total_count - tumor_count,
                'avg_confidence': avg_confidence
            })
            
            current = next_period
        
        return time_series
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            predictions = self.csv_manager.read_csv('predictions.csv')
            training_sessions = self.csv_manager.read_csv('training_sessions.csv')
            
            # Calculate prediction performance
            processing_times = []
            confidences = []
            model_usage = {}
            
            for pred in predictions:
                try:
                    processing_times.append(float(pred['processing_time']))
                    confidences.append(float(pred['confidence']))
                    
                    model = pred['model_used']
                    model_usage[model] = model_usage.get(model, 0) + 1
                except:
                    continue
            
            # Calculate training performance
            training_metrics = []
            for session in training_sessions:
                if session['status'] == 'complete':
                    try:
                        training_metrics.append({
                            'model_name': session['model_name'],
                            'version': session['version'],
                            'accuracy': float(session['accuracy']) if session['accuracy'] else 0.0,
                            'loss': float(session['loss']) if session['loss'] else 0.0,
                            'epochs': int(session['epochs']) if session['epochs'].isdigit() else 0,
                            'start_time': session['start_time'],
                            'end_time': session['end_time']
                        })
                    except:
                        continue
            
            # Get current model detailed metrics
            current_model_metrics = {}
            if self.model_manager.current_model_name:
                try:
                    current_model = self.model_manager.current_model_name
                    if current_model in self.model_manager.models:
                        version_manager = self.model_manager.models[current_model]
                        if version_manager.current_version:
                            current_version = version_manager.versions[version_manager.current_version]
                            metadata = current_version.metadata
                            
                            current_model_metrics = {
                                'model_name': current_model,
                                'version': version_manager.current_version,
                                'accuracy': metadata.get('test_accuracy', 0.0),
                                'loss': metadata.get('test_loss', 0.0),
                                'total_parameters': metadata.get('total_parameters', 0),
                                'input_shape': metadata.get('input_shape'),
                                'training_history': metadata.get('training_history', {})
                            }
                except:
                    pass
            
            return {
                'prediction_performance': {
                    'average_processing_time': round(np.mean(processing_times), 3) if processing_times else 0.0,
                    'average_confidence': round(np.mean(confidences), 3) if confidences else 0.0,
                    'total_predictions': len(predictions),
                    'model_usage': model_usage
                },
                'training_performance': {
                    'completed_sessions': len(training_metrics),
                    'training_history': training_metrics[-5:],  # Last 5 training sessions
                    'average_accuracy': round(np.mean([t['accuracy'] for t in training_metrics]), 3) if training_metrics else 0.0
                },
                'current_model': current_model_metrics
            }
            
        except Exception as e:
            return {
                'error': f'Failed to get performance metrics: {str(e)}',
                'prediction_performance': {},
                'training_performance': {},
                'current_model': {}
            }
    
    def get_recent_activity(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent system activity"""
        try:
            predictions = self.csv_manager.read_csv('predictions.csv')
            training_sessions = self.csv_manager.read_csv('training_sessions.csv')
            dataset = self.csv_manager.read_csv('dataset.csv')
            
            # Combine all activities
            activities = []
            
            # Add recent predictions
            for pred in predictions[-limit:]:
                try:
                    activities.append({
                        'type': 'prediction',
                        'timestamp': pred['timestamp'],
                        'description': f"Prediction completed: {pred['prediction']}",
                        'details': {
                            'confidence': float(pred['confidence']),
                            'model_used': pred['model_used'],
                            'has_tumor': pred['has_tumor'].lower() == 'true'
                        }
                    })
                except:
                    continue
            
            # Add recent training sessions
            for session in training_sessions[-5:]:
                try:
                    activities.append({
                        'type': 'training',
                        'timestamp': session['start_time'],
                        'description': f"Model training {session['status']}: {session['model_name']} v{session['version']}",
                        'details': {
                            'status': session['status'],
                            'epochs': int(session['epochs']) if session['epochs'].isdigit() else 0,
                            'accuracy': float(session['accuracy']) if session['accuracy'] else 0.0
                        }
                    })
                except:
                    continue
            
            # Add recent dataset uploads
            for item in dataset[-5:]:
                try:
                    activities.append({
                        'type': 'dataset',
                        'timestamp': item['upload_date'],
                        'description': f"Dataset image uploaded: {item['filename']}",
                        'details': {
                            'label': item['label'],
                            'size_kb': int(item['size_kb']) if item['size_kb'].isdigit() else 0
                        }
                    })
                except:
                    continue
            
            # Sort by timestamp (newest first)
            activities.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return {
                'activities': activities[:limit],
                'total_activities': len(activities)
            }
            
        except Exception as e:
            return {
                'error': f'Failed to get recent activity: {str(e)}',
                'activities': [],
                'total_activities': 0
            }
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Get comparison data between different models"""
        try:
            training_sessions = self.csv_manager.read_csv('training_sessions.csv')
            predictions = self.csv_manager.read_csv('predictions.csv')
            
            # Group training sessions by model
            model_performance = {}
            
            for session in training_sessions:
                if session['status'] == 'complete':
                    model_name = session['model_name']
                    
                    if model_name not in model_performance:
                        model_performance[model_name] = {
                            'versions': [],
                            'best_accuracy': 0.0,
                            'total_trainings': 0,
                            'prediction_count': 0
                        }
                    
                    try:
                        accuracy = float(session['accuracy']) if session['accuracy'] else 0.0
                        loss = float(session['loss']) if session['loss'] else 0.0
                        
                        model_performance[model_name]['versions'].append({
                            'version': session['version'],
                            'accuracy': accuracy,
                            'loss': loss,
                            'epochs': int(session['epochs']) if session['epochs'].isdigit() else 0,
                            'training_date': session['start_time']
                        })
                        
                        model_performance[model_name]['best_accuracy'] = max(
                            model_performance[model_name]['best_accuracy'], accuracy
                        )
                        model_performance[model_name]['total_trainings'] += 1
                        
                    except:
                        continue
            
            # Add prediction counts
            for pred in predictions:
                model_name = pred['model_used']
                if model_name in model_performance:
                    model_performance[model_name]['prediction_count'] += 1
            
            return {
                'model_comparison': model_performance,
                'total_models': len(model_performance)
            }
            
        except Exception as e:
            return {
                'error': f'Failed to get model comparison: {str(e)}',
                'model_comparison': {},
                'total_models': 0
            }
    
    def update_system_stats(self):
        """Update system statistics (called periodically)"""
        try:
            current_date = datetime.now().strftime('%Y-%m-%d')
            stats = self.get_system_stats()
            
            # Update or append to system_stats.csv
            system_stats_data = self.csv_manager.read_csv('system_stats.csv')
            
            # Check if today's stats already exist
            today_stats = None
            for i, stat in enumerate(system_stats_data):
                if stat['date'] == current_date:
                    today_stats = i
                    break
            
            new_stat_row = {
                'date': current_date,
                'total_scans': str(stats['total_scans']),
                'detected_tumors': str(stats['detected_tumors']),
                'model_accuracy': str(stats['model_accuracy']),
                'active_sessions': '1'  # Simplified - in real app, track actual sessions
            }
            
            if today_stats is not None:
                # Update existing row
                system_stats_data[today_stats] = new_stat_row
                self.csv_manager.write_csv('system_stats.csv', system_stats_data)
            else:
                # Append new row
                self.csv_manager.append_csv('system_stats.csv', new_stat_row)
            
            return True
            
        except Exception as e:
            print(f"Failed to update system stats: {e}")
            return False
