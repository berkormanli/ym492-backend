from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

# Import endpoint creators
from dataset_endpoints import create_dataset_endpoints
from model_endpoints import create_model_endpoints
from prediction_endpoints import create_prediction_endpoints
from dashboard_endpoints import create_dashboard_endpoints

# Import managers
from model_manager import BrainTumorModelManager
from csv_manager import CSVManager

def create_app():
    """Create and configure the Flask application with all endpoints"""
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure app
    app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
    app.config['UPLOAD_FOLDER'] = 'uploads'
    
    # Enable CORS for frontend integration
    CORS(app, origins=['http://localhost:3000', 'http://localhost:3001'])
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    # Create necessary directories
    os.makedirs(os.path.join(BASE_DIR, 'uploads'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'heatmaps'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'thumbnails'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'cache'), exist_ok=True)
    
    # Initialize managers
    model_manager = BrainTumorModelManager()
    csv_manager = CSVManager()
    
    # Try to load existing model
    try:
        latest_model = model_manager.get_latest_version("brain_tumor_model")
        model_manager.load_model(
            "brain_tumor_model",
            f"models/brain_tumor_model/{latest_model}/model.h5",
            latest_model
        )
        print("✅ Existing model loaded successfully")
    except Exception as e:
        print(f"⚠️  No existing model found: {e}")
        print("   Models can be trained using the /api/models/train endpoint")
    
    # Add all endpoint categories
    print("🔧 Setting up endpoints...")
    
    # Category 1: Dataset Management
    app = create_dataset_endpoints(app)
    print("✅ Dataset endpoints added")
    
    # Category 2: Model Management & Training
    app = create_model_endpoints(app, model_manager)
    print("✅ Model endpoints added")
    
    # Category 3: Analysis & Predictions
    app = create_prediction_endpoints(app, model_manager)
    print("✅ Prediction endpoints added")
    
    # Category 4: Dashboard & Analytics
    app = create_dashboard_endpoints(app, model_manager)
    print("✅ Dashboard endpoints added")
    
    # Root endpoints
    @app.route('/')
    def index():
        """API root endpoint"""
        return jsonify({
            'message': 'Brain MRI Tumor Detection API',
            'version': '1.0.0',
            'status': 'online',
            'current_model': model_manager.current_model_name,
            'available_endpoints': {
                'dataset': [
                    'POST /api/dataset/upload',
                    'GET /api/dataset',
                    'GET /api/dataset/{id}',
                    'PUT /api/dataset/{id}',
                    'DELETE /api/dataset/{id}',
                    'GET /api/dataset/stats',
                    'POST /api/dataset/bulk',
                    'GET /api/dataset/{id}/thumbnail',
                    'GET /api/dataset/{id}/image'
                ],
                'models': [
                    'GET /api/models',
                    'POST /api/models/train',
                    'GET /api/models/training/status',
                    'GET /api/models/training/logs',
                    'POST /api/models/training/stop',
                    'GET /api/models/training/sessions',
                    'POST /api/models/switch',
                    'GET /api/models/{name}/versions',
                    'GET /api/models/{name}/metrics',
                    'DELETE /api/models/{name}/{version}',
                    'GET /api/models/current',
                    'POST /api/models/compare'
                ],
                'predictions': [
                    'POST /api/predict',
                    'GET /api/predictions/history',
                    'GET /api/predictions/{id}',
                    'DELETE /api/predictions/{id}',
                    'GET /api/predictions/{id}/image',
                    'GET /api/predictions/{id}/heatmap',
                    'GET /api/predictions/stats',
                    'POST /api/predictions/bulk',
                    'GET /api/predictions/export',
                    'GET /api/predictions/recent',
                    'GET /api/predictions/summary'
                ],
                'dashboard': [
                    'GET /api/dashboard/stats',
                    'GET /api/dashboard/usage',
                    'GET /api/dashboard/performance',
                    'GET /api/dashboard/recent',
                    'GET /api/dashboard/overview',
                    'GET /api/dashboard/models/comparison',
                    'GET /api/dashboard/charts/predictions',
                    'GET /api/dashboard/charts/accuracy',
                    'GET /api/dashboard/export',
                    'GET /api/dashboard/health',
                    'POST /api/dashboard/stats/update'
                ]
            }
        })
    
    @app.route('/api/health')
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': csv_manager.get_current_timestamp(),
            'model_loaded': model_manager.current_model_name is not None,
            'current_model': model_manager.current_model_name
        })
    
    @app.route('/api/info')
    def api_info():
        """API information endpoint"""
        return jsonify({
            'api_name': 'Brain MRI Tumor Detection API',
            'version': '1.0.0',
            'description': 'Complete API for brain tumor detection with model training and dataset management',
            'features': [
                'Image upload and analysis',
                'Model training and version management',
                'Dataset management with thumbnails',
                'Prediction history and analytics',
                'Real-time training monitoring',
                'Comprehensive dashboard analytics',
                'CSV-based data storage',
                'Heatmap generation for tumor regions'
            ],
            'storage': {
                'type': 'CSV files',
                'location': './data/',
                'files': [
                    'dataset.csv - Dataset images metadata',
                    'predictions.csv - Prediction history',
                    'training_sessions.csv - Training session logs',
                    'system_stats.csv - System statistics'
                ]
            },
            'directories': {
                'uploads': 'Dataset images',
                'predictions': 'Prediction images',
                'heatmaps': 'Generated heatmaps',
                'thumbnails': 'Image thumbnails',
                'models': 'Trained model files',
                'data': 'CSV data files'
            }
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(413)
    def file_too_large(error):
        return jsonify({'error': 'File too large. Maximum size is 32MB'}), 413
    
    return app

if __name__ == '__main__':
    print("🚀 Starting Brain MRI Tumor Detection API...")
    print("=" * 50)
    
    app = create_app()
    
    print("=" * 50)
    print("🌐 API Server running at: http://localhost:5000")
    print("📊 Dashboard: Connect your frontend to these endpoints")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000)
