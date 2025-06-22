from flask import Flask, request, jsonify
import json
from csv_manager import CSVManager
from model_manager import BrainTumorModelManager
from model_training_manager import ModelTrainingManager
from model_version_manager import ModelVersionManager

def create_model_endpoints(app: Flask, model_manager: BrainTumorModelManager):
    """Create model management and training endpoints"""
    
    # Initialize managers
    csv_manager = CSVManager()
    training_manager = ModelTrainingManager(csv_manager, model_manager)
    version_manager = ModelVersionManager(csv_manager, model_manager)
    
    @app.route('/api/models', methods=['GET'])
    def get_models():
        """List available models and versions"""
        try:
            models_info = version_manager.get_available_models()
            
            return jsonify({
                'success': True,
                'data': {
                    'models': models_info,
                    'current_model': model_manager.current_model_name,
                    'current_version': version_manager.current_version,
                    'total_models': len(models_info)
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/train', methods=['POST'])
    def start_training():
        """Start model training"""
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['model_name', 'version']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            # Set default values
            training_config = {
                'model_name': data['model_name'],
                'version': data['version'],
                'epochs': data.get('epochs', 20),
                'batch_size': data.get('batch_size', 32),
                'validation_split': data.get('validation_split', 0.2),
                'base_model_type': data.get('base_model_type', 'VGG19'),
                'dataset_size': data.get('dataset_size', 0)
            }
            
            # Start training
            session_id = training_manager.start_training(training_config)
            
            return jsonify({
                'success': True,
                'data': {
                    'session_id': session_id,
                    'message': 'Training started successfully',
                    'config': training_config
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/training/status', methods=['GET'])
    def get_training_status():
        """Get current training status"""
        try:
            status = training_manager.get_training_status()
            
            return jsonify({
                'success': True,
                'data': status
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/training/logs', methods=['GET'])
    def get_training_logs():
        """Get training logs"""
        try:
            session_id = request.args.get('session_id')
            logs = training_manager.get_training_logs(session_id)
            
            return jsonify({
                'success': True,
                'data': {
                    'logs': logs,
                    'session_id': session_id
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/training/stop', methods=['POST'])
    def stop_training():
        """Stop current training"""
        try:
            success = training_manager.stop_training()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Training stop requested'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No active training to stop'
                })
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/training/sessions', methods=['GET'])
    def get_training_sessions():
        """Get all training sessions"""
        try:
            sessions = training_manager.get_training_sessions()
            
            return jsonify({
                'success': True,
                'data': {
                    'sessions': sessions,
                    'total': len(sessions)
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/switch', methods=['POST'])
    def switch_model():
        """Switch active model or version"""
        try:
            data = request.get_json()
            model_name = data.get('model_name')
            version = data.get('version')
            
            if not model_name:
                return jsonify({'error': 'model_name is required'}), 400
            
            success = version_manager.switch_model(model_name, version)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Switched to model: {model_name}' + (f' version: {version}' if version else ''),
                    'current_model': model_manager.current_model_name
                })
            else:
                return jsonify({'error': 'Failed to switch model'}), 400
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/<model_name>/versions', methods=['GET'])
    def get_model_versions(model_name):
        """Get model version history"""
        try:
            versions = version_manager.get_model_versions(model_name)
            
            return jsonify({
                'success': True,
                'data': {
                    'model_name': model_name,
                    'versions': versions,
                    'total_versions': len(versions)
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/<model_name>/metrics', methods=['GET'])
    def get_model_metrics(model_name):
        """Get model performance metrics"""
        try:
            metrics = version_manager.get_model_metrics(model_name)
            
            if not metrics:
                return jsonify({'error': 'Model not found or no metrics available'}), 404
            
            return jsonify({
                'success': True,
                'data': {
                    'model_name': model_name,
                    'metrics': metrics
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/<model_name>/<version>', methods=['DELETE'])
    def delete_model_version(model_name, version):
        """Delete specific model version"""
        try:
            success = version_manager.delete_model_version(model_name, version)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Model version {model_name}:{version} deleted successfully'
                })
            else:
                return jsonify({
                    'error': 'Failed to delete model version. It may be the current version or not exist.'
                }), 400
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/current', methods=['GET'])
    def get_current_model():
        """Get current active model information"""
        try:
            current_model = model_manager.current_model_name
            
            if not current_model:
                return jsonify({
                    'success': True,
                    'data': {
                        'current_model': None,
                        'message': 'No model currently active'
                    }
                })
            
            # Get current model metrics
            metrics = version_manager.get_model_metrics(current_model)
            
            return jsonify({
                'success': True,
                'data': {
                    'current_model': current_model,
                    'metrics': metrics
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/compare', methods=['POST'])
    def compare_models():
        """Compare performance metrics between different models/versions"""
        try:
            data = request.get_json()
            models_to_compare = data.get('models', [])
            
            if len(models_to_compare) < 2:
                return jsonify({'error': 'At least 2 models required for comparison'}), 400
            
            comparison_data = []
            
            for model_info in models_to_compare:
                model_name = model_info.get('model_name')
                version = model_info.get('version')
                
                if not model_name:
                    continue
                
                # Get metrics for this model
                metrics = version_manager.get_model_metrics(model_name)
                
                if metrics:
                    comparison_data.append({
                        'model_name': model_name,
                        'version': version or metrics.get('current_version'),
                        'accuracy': metrics.get('accuracy', 0.0),
                        'loss': metrics.get('loss', 0.0),
                        'total_parameters': metrics.get('total_parameters', 0),
                        'input_shape': metrics.get('input_shape')
                    })
            
            return jsonify({
                'success': True,
                'data': {
                    'comparison': comparison_data,
                    'total_compared': len(comparison_data)
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app
