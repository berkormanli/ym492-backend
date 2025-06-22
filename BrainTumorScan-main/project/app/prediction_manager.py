import os
import json
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image, ImageDraw
import cv2
from csv_manager import CSVManager
from model_manager import BrainTumorModelManager
from pathlib import Path

class PredictionManager:
    def __init__(self, csv_manager: CSVManager, model_manager: BrainTumorModelManager):
        self.csv_manager = csv_manager
        self.model_manager = model_manager
        self.predictions_dir = Path("predictions")
        self.heatmaps_dir = Path("heatmaps")
        self.predictions_dir.mkdir(exist_ok=True)
        self.heatmaps_dir.mkdir(exist_ok=True)
    
    def create_prediction(self, image_path: str, save_image: bool = True) -> Dict[str, Any]:
        """Create a new prediction with enhanced features"""
        start_time = time.time()
        
        try:
            # Generate prediction ID
            prediction_id = self.csv_manager.generate_id("pred-")
            
            # Make prediction using model manager
            prediction_text, confidence = self.model_manager.predict(image_path)
            print(f"Prediction: {prediction_text}, Confidence: {confidence}")
            # Determine if tumor is detected
            #has_tumor = "tumor" in prediction_text.lower() and "no" not in prediction_text.lower()
            tumor_percentage = round((1 - confidence) * 100, 2)
            if confidence <= 0.5:
                has_tumor = True
            else:
                has_tumor = False
            
            # Generate regions (mock implementation - in real scenario, this would come from model)
            regions = self.generate_regions(image_path, has_tumor, tumor_percentage)
            #regions = self._generate_mock_regions(has_tumor, confidence)
            
            # Calculate processing time
            processing_time = round(time.time() - start_time, 2)
            
            # Save prediction image if requested
            prediction_image_path = ""
            heatmap_path = ""
            
            if save_image:
                # Copy original image to predictions directory
                prediction_image_path = self._save_prediction_image(prediction_id, image_path)
                
                # Generate heatmap
                heatmap_path = self._generate_heatmap(prediction_id, image_path, regions, has_tumor)
            
            # Create prediction record
            prediction_record = {
                'id': prediction_id,
                'timestamp': self.csv_manager.get_current_timestamp(),
                'image_path': prediction_image_path,
                'prediction': prediction_text,
                'confidence': str(confidence),
                'has_tumor': str(has_tumor).lower(),
                'tumor_percentage': tumor_percentage,
                'regions': json.dumps(regions),
                'model_used': self.model_manager.current_model_name,
                'processing_time': str(processing_time)
            }
            
            # Save to CSV
            self.csv_manager.append_csv('predictions.csv', prediction_record)
            
            # Return enhanced prediction result
            return {
                'id': prediction_id,
                'prediction': prediction_text,
                'confidence': confidence,
                'hasTumor': has_tumor,
                'tumor_percentage': tumor_percentage,
                'regions': regions,
                'processing_time': processing_time,
                'model_used': self.model_manager.current_model_name,
                'timestamp': prediction_record['timestamp'],
                'image_url': f"/api/predictions/{prediction_id}/image" if prediction_image_path else None,
                'heatmap_url': f"/api/predictions/{prediction_id}/heatmap" if heatmap_path else None
            }
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def generate_regions(self, image_path, has_tumor: bool, confidence: float) -> List[Dict[str, int]]:
        if not has_tumor or confidence < 0.5:
            return []
        
        img = tf.keras.utils.load_img(image_path, target_size=(150, 150), color_mode='grayscale')
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        grad_model = tf.keras.models.Model(
            [self.model_manager.current_model.inputs],
            [self.model_manager.current_model.get_layer("conv2d_1").output, self.model_manager.current_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = 1 - predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :]
        heatmap = tf.reduce_sum(heatmap, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (150, 150))
        heatmap_uint8 = np.uint8(255 * heatmap)

        _, thresh = cv2.threshold(heatmap_uint8, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        orig = cv2.imread(image_path)
        orig = cv2.resize(orig, (150, 150))
        
        regions = []
        """for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            region = {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'confidence': float(predictions[0][0])
            }
            regions.append(region) """

        if contours:
            margin = 10 
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            shrink_ratio = 0.8
            new_w = int(w * shrink_ratio)
            new_h = int(h * shrink_ratio)
            center_x = x + w // 2
            center_y = y + h // 2
            new_x = max(center_x - new_w // 2, 0)
            new_y = max(center_y - new_h // 2, 0)
            regions.append({
                'x': center_x,
                'y': center_y,
                'width': new_w,
                'height': new_h
            })
            # Yeşil kutu
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, orig.shape[1])
            y2 = min(y + h + margin, orig.shape[0])

        return regions

    def _generate_mock_regions(self, has_tumor: bool, confidence: float) -> List[Dict[str, int]]:
        """Generate mock tumor regions (replace with actual detection logic)"""
        if not has_tumor or confidence < 0.5:
            return []
        
        # Generate 1-3 regions based on confidence
        num_regions = min(3, max(1, int(confidence * 3)))
        regions = []
        
        for i in range(num_regions):
            # Generate random regions (in real implementation, these would come from the model)
            region = {
                'x': np.random.randint(50, 150),
                'y': np.random.randint(50, 150),
                'width': np.random.randint(30, 80),
                'height': np.random.randint(30, 80),
                'confidence': round(confidence + np.random.uniform(-0.1, 0.1), 2)
            }
            regions.append(region)
        
        return regions
    
    def _save_prediction_image(self, prediction_id: str, original_path: str) -> str:
        """Save a copy of the prediction image"""
        try:
            # Get file extension
            _, ext = os.path.splitext(original_path)
            if not ext:
                ext = '.jpg'
            
            # Create new filename
            new_filename = f"{prediction_id}{ext}"
            new_path = self.predictions_dir / new_filename
            
            # Copy the image
            with Image.open(original_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                img.save(new_path, 'JPEG', quality=95)
            
            return str(new_path)
            
        except Exception as e:
            print(f"Error saving prediction image: {e}")
            return ""
    
    def _generate_heatmap(self, prediction_id: str, image_path: str, regions: List[Dict], has_tumor: bool) -> str:
        """Generate a heatmap overlay for the prediction"""
        try:
            if not has_tumor or not regions:
                return ""
            
            # Load the original image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Create a copy for the heatmap
                heatmap_img = img.copy()
                draw = ImageDraw.Draw(heatmap_img, 'RGBA')
                
                # Draw regions
                for region in regions:
                    x = region['x']
                    y = region['y']
                    width = region['width']
                    height = region['height']
                    confidence = region.get('confidence', 0.5)
                    
                    # Calculate color intensity based on confidence
                    alpha = int(confidence * 100)
                    color = (255, 0, 0, alpha)  # Red with varying transparency
                    
                    # Draw rectangle
                    draw.rectangle(
                        [x, y, x + width, y + height],
                        fill=color,
                        outline=(255, 0, 0, 255),
                        width=2
                    )
                
                # Save heatmap
                heatmap_filename = f"{prediction_id}_heatmap.jpg"
                heatmap_path = self.heatmaps_dir / heatmap_filename
                heatmap_img.save(heatmap_path, 'JPEG', quality=95)
                
                return str(heatmap_path)
                
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            return ""
    
    def get_prediction_history(self, page: int = 1, per_page: int = 10, 
                             model_filter: str = "", date_filter: str = "") -> Dict[str, Any]:
        """Get prediction history with filtering and pagination"""
        predictions = self.csv_manager.read_csv('predictions.csv')
        
        # Apply filters
        if model_filter:
            predictions = [p for p in predictions if p['model_used'] == model_filter]
        
        if date_filter:
            # Filter by date (YYYY-MM-DD format)
            predictions = [p for p in predictions if p['timestamp'].startswith(date_filter)]
        
        # Sort by timestamp (newest first)
        predictions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Convert string fields to appropriate types
        for pred in predictions:
            try:
                pred['confidence'] = float(pred['confidence'])
                pred['has_tumor'] = pred['has_tumor'].lower() == 'true'
                pred['regions'] = json.loads(pred['regions']) if pred['regions'] else []
                pred['processing_time'] = float(pred['processing_time'])
            except:
                pass
        
        # Calculate pagination
        total_items = len(predictions)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_predictions = predictions[start_idx:end_idx]
        
        # Add URLs to predictions
        for pred in paginated_predictions:
            pred['image_url'] = f"/api/predictions/{pred['id']}/image"
            pred['heatmap_url'] = f"/api/predictions/{pred['id']}/heatmap"
        
        return {
            'predictions': paginated_predictions,
            'total': total_items,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_items + per_page - 1) // per_page
        }
    
    def get_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific prediction"""
        predictions = self.csv_manager.read_csv('predictions.csv')
        
        for pred in predictions:
            if pred['id'] == prediction_id:
                # Convert string fields to appropriate types
                try:
                    pred['confidence'] = float(pred['confidence'])
                    pred['has_tumor'] = pred['has_tumor'].lower() == 'true'
                    pred['regions'] = json.loads(pred['regions']) if pred['regions'] else []
                    pred['processing_time'] = float(pred['processing_time'])
                except:
                    pass
                
                # Add URLs
                pred['image_url'] = f"/api/predictions/{pred['id']}/image"
                pred['heatmap_url'] = f"/api/predictions/{pred['id']}/heatmap"
                
                return pred
        
        return None
    
    def delete_prediction(self, prediction_id: str) -> bool:
        """Delete a prediction and its associated files"""
        try:
            # Get prediction details first
            prediction = self.get_prediction(prediction_id)
            if not prediction:
                return False
            
            # Delete associated files
            image_path = self.predictions_dir / f"{prediction_id}.jpg"
            heatmap_path = self.heatmaps_dir / f"{prediction_id}_heatmap.jpg"
            
            if image_path.exists():
                os.remove(image_path)
            
            if heatmap_path.exists():
                os.remove(heatmap_path)
            
            # Delete from CSV
            self.csv_manager.delete_csv_row('predictions.csv', prediction_id)
            
            return True
            
        except Exception:
            return False
    
    def get_prediction_image_path(self, prediction_id: str) -> Optional[str]:
        """Get the file path for a prediction image"""
        prediction = self.get_prediction(prediction_id)
        if not prediction:
            return None
        
        image_path = self.predictions_dir / f"{prediction_id}.jpg"
        if image_path.exists():
            return str(image_path)
        
        return None
    
    def get_prediction_heatmap_path(self, prediction_id: str) -> Optional[str]:
        """Get the file path for a prediction heatmap"""
        prediction = self.get_prediction(prediction_id)
        if not prediction:
            return None
        
        heatmap_path = self.heatmaps_dir / f"{prediction_id}_heatmap.jpg"
        if heatmap_path.exists():
            return str(heatmap_path)
        
        return None
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        predictions = self.csv_manager.read_csv('predictions.csv')
        
        total_predictions = len(predictions)
        tumor_predictions = len([p for p in predictions if p['has_tumor'].lower() == 'true'])
        no_tumor_predictions = total_predictions - tumor_predictions
        
        # Calculate average confidence
        confidences = []
        processing_times = []
        
        for pred in predictions:
            try:
                confidences.append(float(pred['confidence']))
                processing_times.append(float(pred['processing_time']))
            except:
                pass
        
        avg_confidence = round(np.mean(confidences), 3) if confidences else 0.0
        avg_processing_time = round(np.mean(processing_times), 3) if processing_times else 0.0
        
        # Get model usage stats
        model_usage = {}
        for pred in predictions:
            model = pred['model_used']
            model_usage[model] = model_usage.get(model, 0) + 1
        
        return {
            'total_predictions': total_predictions,
            'tumor_detected': tumor_predictions,
            'no_tumor_detected': no_tumor_predictions,
            'tumor_detection_rate': round((tumor_predictions / total_predictions * 100) if total_predictions > 0 else 0, 1),
            'average_confidence': avg_confidence,
            'average_processing_time': avg_processing_time,
            'model_usage': model_usage
        }
    
    def bulk_delete_predictions(self, prediction_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple predictions"""
        deleted_count = 0
        failed_count = 0
        
        for prediction_id in prediction_ids:
            if self.delete_prediction(prediction_id):
                deleted_count += 1
            else:
                failed_count += 1
        
        return {
            'deleted': deleted_count,
            'failed': failed_count,
            'total': len(prediction_ids)
        }
