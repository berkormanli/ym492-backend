from flask import Flask, request, jsonify, send_file
import json
from werkzeug.utils import secure_filename
import os
import tempfile
from csv_manager import CSVManager
from model_manager import BrainTumorModelManager
from prediction_manager import PredictionManager

def create_prediction_endpoints(app: Flask, model_manager: BrainTumorModelManager):
    """Create prediction and analysis endpoints"""
    
    # Initialize managers
    csv_manager = CSVManager()
    prediction_manager = PredictionManager(csv_manager, model_manager)
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/api/predict', methods=['POST'])
    def predict_image():
        """Enhanced image analysis endpoint"""
        try:
            # Check if file is provided
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Get options
            save_to_history = request.form.get('save_to_history', 'true').lower() == 'true'
            
            # Save file temporarily
            filename = secure_filename(file.filename)
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, filename)
            file.save(temp_path)
            
            try:
                # Create prediction
                result = prediction_manager.create_prediction(temp_path, save_image=save_to_history)
                
                return jsonify({
                    'success': True,
                    'data': result
                })
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                os.rmdir(temp_dir)
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/history', methods=['GET'])
    def get_prediction_history():
        """Get prediction history with filtering and pagination"""
        try:
            # Get query parameters
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 10))
            model_filter = request.args.get('model', '')
            date_filter = request.args.get('date', '')
            
            # Get filtered and paginated history
            result = prediction_manager.get_prediction_history(
                page=page,
                per_page=per_page,
                model_filter=model_filter,
                date_filter=date_filter
            )
            
            return jsonify({
                'success': True,
                'data': result
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/<prediction_id>', methods=['GET'])
    def get_prediction_details(prediction_id):
        """Get specific prediction details"""
        try:
            prediction = prediction_manager.get_prediction(prediction_id)
            
            if prediction:
                return jsonify({
                    'success': True,
                    'data': prediction
                })
            else:
                return jsonify({'error': 'Prediction not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/<prediction_id>', methods=['DELETE'])
    def delete_prediction(prediction_id):
        """Delete a prediction from history"""
        try:
            success = prediction_manager.delete_prediction(prediction_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Prediction deleted successfully'
                })
            else:
                return jsonify({'error': 'Failed to delete prediction or prediction not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/<prediction_id>/image', methods=['GET'])
    def get_prediction_image(prediction_id):
        """Get the original image for a prediction"""
        try:
            image_path = prediction_manager.get_prediction_image_path(prediction_id)
            if image_path and os.path.exists(os.path.abspath(os.path.normpath(image_path))):
                return send_file(os.path.abspath(os.path.normpath(image_path)), mimetype='image/jpeg')
            else:
                return jsonify({'error': 'Image not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/<prediction_id>/heatmap', methods=['GET'])
    def get_prediction_heatmap(prediction_id):
        """Get the heatmap overlay for a prediction"""
        try:
            heatmap_path = prediction_manager.get_prediction_heatmap_path(prediction_id)
            
            if heatmap_path and os.path.exists(heatmap_path):
                return send_file(heatmap_path, mimetype='image/jpeg')
            else:
                return jsonify({'error': 'Heatmap not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/stats', methods=['GET'])
    def get_prediction_stats():
        """Get prediction statistics"""
        try:
            stats = prediction_manager.get_prediction_stats()
            
            return jsonify({
                'success': True,
                'data': stats
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/bulk', methods=['POST'])
    def bulk_delete_predictions():
        """Bulk delete predictions"""
        try:
            data = request.get_json()
            prediction_ids = data.get('prediction_ids', [])
            
            if not prediction_ids:
                return jsonify({'error': 'No prediction IDs provided'}), 400
            
            result = prediction_manager.bulk_delete_predictions(prediction_ids)
            
            return jsonify({
                'success': True,
                'data': result
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/export', methods=['GET'])
    def export_predictions():
        """Export prediction history as CSV"""
        try:
            # Get query parameters for filtering
            model_filter = request.args.get('model', '')
            date_filter = request.args.get('date', '')
            
            # Get all predictions (no pagination for export)
            result = prediction_manager.get_prediction_history(
                page=1,
                per_page=10000,  # Large number to get all
                model_filter=model_filter,
                date_filter=date_filter
            )
            
            predictions = result['predictions']
            
            # Create CSV content
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'ID', 'Timestamp', 'Prediction', 'Confidence', 'Has Tumor',
                'Model Used', 'Processing Time', 'Regions Count'
            ])
            
            # Write data
            for pred in predictions:
                writer.writerow([
                    pred['id'],
                    pred['timestamp'],
                    pred['prediction'],
                    pred['confidence'],
                    pred['has_tumor'],
                    pred['model_used'],
                    pred['processing_time'],
                    len(pred['regions'])
                ])
            
            # Create response
            output.seek(0)
            
            from flask import Response
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=predictions_export.csv'}
            )
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/recent', methods=['GET'])
    def get_recent_predictions():
        """Get recent predictions for dashboard"""
        try:
            limit = int(request.args.get('limit', 5))
            
            # Get recent predictions
            result = prediction_manager.get_prediction_history(
                page=1,
                per_page=limit
            )
            
            return jsonify({
                'success': True,
                'data': {
                    'recent_predictions': result['predictions'],
                    'total_count': result['total']
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predictions/summary', methods=['GET'])
    def get_predictions_summary():
        """Get prediction summary for a specific time period"""
        try:
            # Get query parameters
            period = request.args.get('period', 'week')  # day, week, month, year
            
            # This is a simplified implementation
            # In a real application, you'd filter by actual date ranges
            stats = prediction_manager.get_prediction_stats()
            
            # Add period-specific data (mock implementation)
            period_data = {
                'period': period,
                'total_predictions': stats['total_predictions'],
                'tumor_detected': stats['tumor_detected'],
                'accuracy_trend': [0.92, 0.94, 0.93, 0.95, 0.94],  # Mock trend data
                'volume_trend': [12, 15, 18, 14, 16]  # Mock volume data
            }
            
            return jsonify({
                'success': True,
                'data': {
                    'summary': period_data,
                    'stats': stats
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app
