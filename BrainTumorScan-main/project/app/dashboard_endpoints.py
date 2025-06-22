from flask import Flask, request, jsonify
from csv_manager import CSVManager
from model_manager import BrainTumorModelManager
from analytics_manager import AnalyticsManager

def create_dashboard_endpoints(app: Flask, model_manager: BrainTumorModelManager):
    """Create dashboard and analytics endpoints"""
    
    # Initialize managers
    csv_manager = CSVManager()
    analytics_manager = AnalyticsManager(csv_manager, model_manager)
    
    @app.route('/api/dashboard/stats', methods=['GET'])
    def get_dashboard_stats():
        """Get overall system statistics for dashboard"""
        try:
            stats = analytics_manager.get_system_stats()
            
            return jsonify({
                'success': True,
                'data': stats
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/usage', methods=['GET'])
    def get_usage_analytics():
        """Get usage analytics over time"""
        try:
            period = request.args.get('period', 'week')  # day, week, month
            
            if period not in ['day', 'week', 'month']:
                return jsonify({'error': 'Invalid period. Use: day, week, or month'}), 400
            
            analytics = analytics_manager.get_usage_analytics(period)
            
            return jsonify({
                'success': True,
                'data': analytics
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/performance', methods=['GET'])
    def get_performance_metrics():
        """Get model performance metrics"""
        try:
            metrics = analytics_manager.get_performance_metrics()
            
            return jsonify({
                'success': True,
                'data': metrics
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/recent', methods=['GET'])
    def get_recent_activity():
        """Get recent system activity"""
        try:
            limit = int(request.args.get('limit', 10))
            
            if limit > 50:  # Prevent excessive data
                limit = 50
            
            activity = analytics_manager.get_recent_activity(limit)
            
            return jsonify({
                'success': True,
                'data': activity
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/overview', methods=['GET'])
    def get_dashboard_overview():
        """Get comprehensive dashboard overview"""
        try:
            # Get all dashboard data in one call for efficiency
            stats = analytics_manager.get_system_stats()
            recent_activity = analytics_manager.get_recent_activity(5)
            performance = analytics_manager.get_performance_metrics()
            
            # Get quick usage analytics for the week
            usage = analytics_manager.get_usage_analytics('week')
            
            overview = {
                'system_stats': stats,
                'recent_activity': recent_activity['activities'][:5],
                'performance_summary': {
                    'current_model': performance.get('current_model', {}),
                    'avg_processing_time': performance.get('prediction_performance', {}).get('average_processing_time', 0.0),
                    'avg_confidence': performance.get('prediction_performance', {}).get('average_confidence', 0.0)
                },
                'usage_trend': usage.get('trends', {}).get('total_predictions', [])[-7:],  # Last 7 data points
                'last_updated': analytics_manager.csv_manager.get_current_timestamp()
            }
            
            return jsonify({
                'success': True,
                'data': overview
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/models/comparison', methods=['GET'])
    def get_model_comparison():
        """Get model comparison data"""
        try:
            comparison = analytics_manager.get_model_comparison()
            
            return jsonify({
                'success': True,
                'data': comparison
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/charts/predictions', methods=['GET'])
    def get_predictions_chart_data():
        """Get chart data for predictions over time"""
        try:
            period = request.args.get('period', 'week')
            chart_type = request.args.get('type', 'line')  # line, bar, pie
            
            usage_data = analytics_manager.get_usage_analytics(period)
            
            if chart_type == 'pie':
                # Pie chart data for tumor vs no tumor
                total_tumor = sum(usage_data.get('trends', {}).get('tumor_detections', []))
                total_predictions = sum(usage_data.get('trends', {}).get('total_predictions', []))
                total_no_tumor = total_predictions - total_tumor
                
                chart_data = {
                    'type': 'pie',
                    'data': [
                        {'label': 'Tumor Detected', 'value': total_tumor},
                        {'label': 'No Tumor', 'value': total_no_tumor}
                    ]
                }
            else:
                # Line/Bar chart data
                chart_data = {
                    'type': chart_type,
                    'labels': [point['label'] for point in usage_data.get('time_series', [])],
                    'datasets': [
                        {
                            'label': 'Total Predictions',
                            'data': usage_data.get('trends', {}).get('total_predictions', []),
                            'color': '#3B82F6'
                        },
                        {
                            'label': 'Tumor Detections',
                            'data': usage_data.get('trends', {}).get('tumor_detections', []),
                            'color': '#EF4444'
                        }
                    ]
                }
            
            return jsonify({
                'success': True,
                'data': chart_data
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/charts/accuracy', methods=['GET'])
    def get_accuracy_chart_data():
        """Get chart data for model accuracy over time"""
        try:
            training_sessions = analytics_manager.csv_manager.read_csv('training_sessions.csv')
            
            # Filter completed sessions and sort by date
            completed_sessions = []
            for session in training_sessions:
                if session['status'] == 'complete':
                    try:
                        completed_sessions.append({
                            'model_name': session['model_name'],
                            'version': session['version'],
                            'accuracy': float(session['accuracy']) if session['accuracy'] else 0.0,
                            'loss': float(session['loss']) if session['loss'] else 0.0,
                            'date': session['start_time'][:10]  # Extract date part
                        })
                    except:
                        continue
            
            # Sort by date
            completed_sessions.sort(key=lambda x: x['date'])
            
            chart_data = {
                'type': 'line',
                'labels': [f"{s['model_name']} v{s['version']}" for s in completed_sessions[-10:]],
                'datasets': [
                    {
                        'label': 'Accuracy',
                        'data': [s['accuracy'] for s in completed_sessions[-10:]],
                        'color': '#10B981'
                    },
                    {
                        'label': 'Loss',
                        'data': [s['loss'] for s in completed_sessions[-10:]],
                        'color': '#F59E0B'
                    }
                ]
            }
            
            return jsonify({
                'success': True,
                'data': chart_data
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/export', methods=['GET'])
    def export_dashboard_data():
        """Export dashboard data as JSON"""
        try:
            export_type = request.args.get('type', 'overview')  # overview, detailed, analytics
            
            if export_type == 'detailed':
                # Export detailed analytics
                data = {
                    'system_stats': analytics_manager.get_system_stats(),
                    'usage_analytics': analytics_manager.get_usage_analytics('month'),
                    'performance_metrics': analytics_manager.get_performance_metrics(),
                    'model_comparison': analytics_manager.get_model_comparison(),
                    'recent_activity': analytics_manager.get_recent_activity(50),
                    'export_timestamp': analytics_manager.csv_manager.get_current_timestamp()
                }
            elif export_type == 'analytics':
                # Export analytics data only
                data = {
                    'usage_analytics_week': analytics_manager.get_usage_analytics('week'),
                    'usage_analytics_month': analytics_manager.get_usage_analytics('month'),
                    'performance_metrics': analytics_manager.get_performance_metrics(),
                    'export_timestamp': analytics_manager.csv_manager.get_current_timestamp()
                }
            else:
                # Export overview data
                data = {
                    'system_stats': analytics_manager.get_system_stats(),
                    'recent_activity': analytics_manager.get_recent_activity(10),
                    'current_model_performance': analytics_manager.get_performance_metrics().get('current_model', {}),
                    'export_timestamp': analytics_manager.csv_manager.get_current_timestamp()
                }
            
            return jsonify({
                'success': True,
                'export_type': export_type,
                'data': data
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/health', methods=['GET'])
    def get_system_health():
        """Get system health status"""
        try:
            # Check various system components
            health_status = {
                'overall_status': 'healthy',
                'components': {
                    'model_manager': 'healthy' if model_manager.current_model_name else 'warning',
                    'csv_storage': 'healthy',
                    'prediction_system': 'healthy',
                    'training_system': 'healthy'
                },
                'warnings': [],
                'errors': []
            }
            
            # Check if model is loaded
            if not model_manager.current_model_name:
                health_status['warnings'].append('No model currently active')
                health_status['components']['model_manager'] = 'warning'
            
            # Check CSV files
            try:
                predictions = analytics_manager.csv_manager.read_csv('predictions.csv')
                dataset = analytics_manager.csv_manager.read_csv('dataset.csv')
                training_sessions = analytics_manager.csv_manager.read_csv('training_sessions.csv')
                
                # Check if we have data
                if len(predictions) == 0:
                    health_status['warnings'].append('No predictions recorded yet')
                
                if len(dataset) == 0:
                    health_status['warnings'].append('No dataset images uploaded yet')
                
            except Exception as e:
                health_status['errors'].append(f'CSV storage error: {str(e)}')
                health_status['components']['csv_storage'] = 'error'
            
            # Determine overall status
            if health_status['errors']:
                health_status['overall_status'] = 'error'
            elif health_status['warnings']:
                health_status['overall_status'] = 'warning'
            
            health_status['last_check'] = analytics_manager.csv_manager.get_current_timestamp()
            
            return jsonify({
                'success': True,
                'data': health_status
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/stats/update', methods=['POST'])
    def update_system_stats():
        """Manually trigger system stats update"""
        try:
            success = analytics_manager.update_system_stats()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'System stats updated successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Failed to update system stats'
                }), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app
