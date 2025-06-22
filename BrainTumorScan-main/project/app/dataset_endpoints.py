from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
from PIL import Image
import io
from csv_manager import CSVManager, DatasetManager

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    
def create_dataset_endpoints(app: Flask):
    """Create dataset management endpoints"""
    
    # Initialize managers
    csv_manager = CSVManager()
    dataset_manager = DatasetManager(csv_manager)
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def create_thumbnail(image_path: str, thumbnail_path: str, size: tuple = (150, 150)):
        """Create a thumbnail for an image"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85)
                return True
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return False
    
    @app.route('/api/dataset/upload', methods=['POST'])
    def upload_dataset_images():
        """Upload MRI images with labels to the dataset"""
        try:
            if 'files' not in request.files:
                return jsonify({'error': 'No files provided'}), 400
            
            files = request.files.getlist('files')
            labels = request.form.get('labels')
            notes = request.form.getlist('notes')
            
            uploaded_items = []
            failed_uploads = []
            
            for i, file in enumerate(files):
                if file and file.filename and allowed_file(file.filename):
                    try:
                        # Secure the filename
                        filename = secure_filename(file.filename)
                        
                        # Create unique filename to avoid conflicts
                        base_name, ext = os.path.splitext(filename)
                        unique_filename = f"{base_name}_{csv_manager.generate_id()}_{ext}"
                        
                        # Save the file
                        file_path = os.path.join('uploads', unique_filename)
                        file.save(file_path)
                        
                        # Get file size
                        file_size_kb = os.path.getsize(file_path) // 1024
                        
                        # Create thumbnail
                        thumbnail_filename = f"thumb_{unique_filename}.jpg"
                        thumbnail_path = os.path.join('thumbnails', thumbnail_filename)
                        
                        if create_thumbnail(file_path, thumbnail_path):
                            thumbnail_rel_path = thumbnail_path
                        else:
                            thumbnail_rel_path = ""
                        
                        # Add to dataset
                        item_id = dataset_manager.add_dataset_item(
                            filename=filename,
                            label=labels,
                            file_path=file_path,
                            size_kb=file_size_kb,
                            notes=notes[i] if i < len(notes) else "",
                            thumbnail_path=thumbnail_rel_path
                        )
                        
                        uploaded_items.append({
                            'id': item_id,
                            'filename': filename,
                            'label': labels,
                            'size_kb': file_size_kb
                        })
                        
                    except Exception as e:
                        failed_uploads.append({
                            'filename': file.filename,
                            'error': str(e)
                        })
                else:
                    failed_uploads.append({
                        'filename': file.filename if file else 'Unknown',
                        'error': 'Invalid file type or no file provided'
                    })
            
            return jsonify({
                'success': True,
                'uploaded': uploaded_items,
                'failed': failed_uploads,
                'total_uploaded': len(uploaded_items),
                'total_failed': len(failed_uploads)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dataset', methods=['GET'])
    def get_dataset():
        """Get dataset items with filtering and pagination"""
        try:
            # Get query parameters
            search_query = request.args.get('search', '')
            label_filter = request.args.get('label', '')
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 10))
            
            # Get filtered and paginated data
            result = dataset_manager.get_dataset_items(
                search_query=search_query,
                label_filter=label_filter,
                page=page,
                per_page=per_page
            )
            
            return jsonify({
                'success': True,
                'data': result
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dataset/<item_id>', methods=['GET'])
    def get_dataset_item(item_id):
        """Get a specific dataset item"""
        try:
            item = dataset_manager.get_dataset_item(item_id)
            if item:
                return jsonify({
                    'success': True,
                    'data': item
                })
            else:
                return jsonify({'error': 'Item not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dataset/<item_id>', methods=['PUT'])
    def update_dataset_item(item_id):
        """Update a dataset item"""
        try:
            data = request.get_json()
            
            # Only allow updating certain fields
            allowed_fields = ['label', 'notes']
            updates = {k: v for k, v in data.items() if k in allowed_fields}
            
            if dataset_manager.update_dataset_item(item_id, updates):
                return jsonify({
                    'success': True,
                    'message': 'Item updated successfully'
                })
            else:
                return jsonify({'error': 'Failed to update item'}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dataset/<item_id>', methods=['DELETE'])
    def delete_dataset_item(item_id):
        """Delete a dataset item"""
        try:
            if dataset_manager.delete_dataset_item(item_id):
                return jsonify({
                    'success': True,
                    'message': 'Item deleted successfully'
                })
            else:
                return jsonify({'error': 'Failed to delete item'}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dataset/stats', methods=['GET'])
    def get_dataset_stats():
        """Get dataset statistics"""
        try:
            stats = dataset_manager.get_dataset_stats()
            return jsonify({
                'success': True,
                'data': stats
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dataset/bulk', methods=['POST'])
    def bulk_delete_dataset():
        """Bulk delete dataset items"""
        try:
            data = request.get_json()
            item_ids = data.get('item_ids', [])
            
            if not item_ids:
                return jsonify({'error': 'No item IDs provided'}), 400
            
            result = dataset_manager.bulk_delete(item_ids)
            
            return jsonify({
                'success': True,
                'data': result
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dataset/<item_id>/thumbnail', methods=['GET'])
    def get_dataset_thumbnail(item_id):
        """Get thumbnail for a dataset item"""
        try:
            item = dataset_manager.get_dataset_item(item_id)
            if not item:
                return jsonify({'error': 'Item not found'}), 404
            
            thumbnail_path = os.path.join(BASE_DIR,item.get('thumbnail_path'))
            if thumbnail_path:
                return send_file(thumbnail_path)
            else:
                # Return a placeholder image or 404
                return jsonify({'error': 'Thumbnail not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dataset/<item_id>/image', methods=['GET'])
    def get_dataset_image(item_id):
        """Get the full image for a dataset item"""
        try:
            item = dataset_manager.get_dataset_item(item_id)
            if not item:
                return jsonify({'error': 'Item not found'}), 404
            
            file_path = os.path.join(BASE_DIR, item.get('file_path'))
            
            if file_path:
                # Determine mimetype based on file extension
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.jpg', '.jpeg']:
                    mimetype = 'image/jpeg'
                elif ext == '.png':
                    mimetype = 'image/png'
                elif ext in ['.dcm', '.dicom']:
                    mimetype = 'application/dicom'
                else:
                    mimetype = 'application/octet-stream'
                
                return send_file(file_path, mimetype=mimetype)
            else:
                return jsonify({'error': 'Image file not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app
