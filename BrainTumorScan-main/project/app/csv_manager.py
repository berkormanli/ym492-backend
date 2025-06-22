import csv
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
from pathlib import Path

class CSVManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize CSV files if they don't exist
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        csv_files = {
            'dataset.csv': ['id', 'filename', 'label', 'upload_date', 'size_kb', 'notes', 'file_path', 'thumbnail_path'],
            'predictions.csv': ['id', 'timestamp', 'image_path', 'prediction', 'confidence', 'has_tumor', 'regions', 'model_used', 'processing_time'],
            'training_sessions.csv': ['id', 'model_name', 'version', 'start_time', 'end_time', 'status', 'progress', 'epochs', 'accuracy', 'loss', 'dataset_size', 'logs'],
            'system_stats.csv': ['date', 'total_scans', 'detected_tumors', 'model_accuracy', 'active_sessions']
        }
        
        for filename, headers in csv_files.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
    
    def read_csv(self, filename: str) -> List[Dict[str, Any]]:
        """Read CSV file and return list of dictionaries"""
        file_path = self.data_dir / filename
        if not file_path.exists():
            return []
        
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def write_csv(self, filename: str, data: List[Dict[str, Any]]):
        """Write list of dictionaries to CSV file"""
        if not data:
            return
        
        file_path = self.data_dir / filename
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    
    def append_csv(self, filename: str, row: Dict[str, Any]):
        """Append a single row to CSV file"""
        file_path = self.data_dir / filename
        file_exists = file_path.exists()
        
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    
    def update_csv_row(self, filename: str, row_id: str, updated_data: Dict[str, Any], id_field: str = 'id'):
        """Update a specific row in CSV file"""
        data = self.read_csv(filename)
        for i, row in enumerate(data):
            if row[id_field] == row_id:
                data[i].update(updated_data)
                break
        self.write_csv(filename, data)
    
    def delete_csv_row(self, filename: str, row_id: str, id_field: str = 'id'):
        """Delete a specific row from CSV file"""
        data = self.read_csv(filename)
        data = [row for row in data if row[id_field] != row_id]
        self.write_csv(filename, data)
    
    def generate_id(self, prefix: str = "") -> str:
        """Generate a unique ID"""
        return f"{prefix}{uuid.uuid4().hex[:8]}"
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()

class DatasetManager:
    def __init__(self, csv_manager: CSVManager):
        self.csv_manager = csv_manager
        self.uploads_dir = Path("uploads")
        self.thumbnails_dir = Path("thumbnails")
        self.uploads_dir.mkdir(exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)
    
    def add_dataset_item(self, filename: str, label: str, file_path: str, 
                        size_kb: int, notes: str = "", thumbnail_path: str = "") -> str:
        """Add a new item to the dataset"""
        item_id = self.csv_manager.generate_id("img-")
        print('Adding dataset item with ID:', item_id, label)
        dataset_item = {
            'id': item_id,
            'filename': filename,
            'label': label,
            'upload_date': self.csv_manager.get_current_timestamp(),
            'size_kb': str(size_kb),
            'notes': notes,
            'file_path': file_path,
            'thumbnail_path': thumbnail_path
        }
        
        self.csv_manager.append_csv('dataset.csv', dataset_item)
        return item_id
    
    def get_dataset_items(self, search_query: str = "", label_filter: str = "", 
                         page: int = 1, per_page: int = 10) -> Dict[str, Any]:
        """Get dataset items with filtering and pagination"""
        data = self.csv_manager.read_csv('dataset.csv')
        
        # Apply filters
        if search_query:
            data = [item for item in data if search_query.lower() in item['filename'].lower()]
        
        if label_filter:
            data = [item for item in data if item['label'] == label_filter]
        
        # Calculate pagination
        total_items = len(data)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_data = data[start_idx:end_idx]
        
        return {
            'items': paginated_data,
            'total': total_items,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_items + per_page - 1) // per_page
        }
    
    def get_dataset_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific dataset item"""
        data = self.csv_manager.read_csv('dataset.csv')
        for item in data:
            if item['id'] == item_id:
                return item
        return None
    
    def update_dataset_item(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update a dataset item"""
        try:
            self.csv_manager.update_csv_row('dataset.csv', item_id, updates)
            return True
        except Exception:
            return False
    
    def delete_dataset_item(self, item_id: str) -> bool:
        """Delete a dataset item"""
        try:
            # Get item details first to delete files
            item = self.get_dataset_item(item_id)
            if item:
                # Delete actual files
                if item['file_path'] and os.path.exists(item['file_path']):
                    os.remove(item['file_path'])
                if item['thumbnail_path'] and os.path.exists(item['thumbnail_path']):
                    os.remove(item['thumbnail_path'])
            
            self.csv_manager.delete_csv_row('dataset.csv', item_id)
            return True
        except Exception:
            return False
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        data = self.csv_manager.read_csv('dataset.csv')
        
        total_images = len(data)
        tumor_count = len([item for item in data if item['label'] == 'tumor'])
        no_tumor_count = len([item for item in data if item['label'] == 'no_tumor'])
        
        total_size_kb = sum(int(item['size_kb']) for item in data if item['size_kb'].isdigit())
        
        return {
            'total_images': total_images,
            'tumor_images': tumor_count,
            'no_tumor_images': no_tumor_count,
            'total_size_mb': round(total_size_kb / 1024, 2),
            'distribution': {
                'tumor': round((tumor_count / total_images * 100) if total_images > 0 else 0, 1),
                'no_tumor': round((no_tumor_count / total_images * 100) if total_images > 0 else 0, 1)
            }
        }
    
    def bulk_delete(self, item_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple dataset items"""
        deleted_count = 0
        failed_count = 0
        
        for item_id in item_ids:
            if self.delete_dataset_item(item_id):
                deleted_count += 1
            else:
                failed_count += 1
        
        return {
            'deleted': deleted_count,
            'failed': failed_count,
            'total': len(item_ids)
        }
