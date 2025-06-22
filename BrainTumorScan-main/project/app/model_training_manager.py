import threading
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from csv_manager import CSVManager
from model_manager import BrainTumorModelManager
from model_version_manager import ModelVersionManager
import os
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

class ModelTrainingManager:
    def __init__(self, csv_manager: CSVManager, model_manager: BrainTumorModelManager):
        self.csv_manager = csv_manager
        self.model_manager = model_manager
        self.current_training = None
        self.training_thread = None
        self.training_logs = []
        self.training_status = "idle"
        self.training_progress = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.training_metrics = {}
        
    def start_training(self, training_config: Dict[str, Any]) -> str:
        """Start a new model training session"""
        if self.training_status not in ["idle", "complete", "failed"]:
            raise Exception("Training is already in progress")
        
        # Generate training session ID
        session_id = self.csv_manager.generate_id("train-")
        
        # Create training session record
        training_session = {
            'id': session_id,
            'model_name': training_config.get('model_name', 'brain_tumor_model'),
            'version': training_config.get('version', '1.0.0'),
            'start_time': self.csv_manager.get_current_timestamp(),
            'end_time': '',
            'status': 'preparing',
            'progress': '0',
            'epochs': str(training_config.get('epochs', 20)),
            'accuracy': '',
            'loss': '',
            'dataset_size': str(training_config.get('dataset_size', 0)),
            'logs': json.dumps([])
        }
        
        # Save to CSV
        self.csv_manager.append_csv('training_sessions.csv', training_session)
        
        # Start training in background thread
        self.current_training = session_id
        self.training_status = "preparing"
        self.training_progress = 0
        self.training_logs = []
        self.current_epoch = 0
        self.total_epochs = training_config.get('epochs', 20)
        
        self.training_thread = threading.Thread(
            target=self._run_training,
            args=(session_id, training_config)
        )
        self.training_thread.start()
        
        return session_id
    
    def _run_training(self, session_id: str, config: Dict[str, Any]):
        """Run the actual training process using dataset.csv for image paths and labels"""
        try:
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            from keras.models import Sequential
            from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
            from keras.regularizers import l2
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from sklearn.metrics import classification_report
            from PIL import Image, ImageOps
            import os
            import random

            self._log_message("Initializing training environment...")
            self._update_status("preparing", 5)

            # Get config values early
            model_name = config.get('model_name', 'brain_tumor_model')
            version = config.get('version', '1.0.0')
            parent_version = config.get('parent_version', None)
            batch_size = config.get('batch_size', 64)
            epochs = config.get('epochs', 20)

            # 1. Load dataset.csv
            self._log_message("Loading dataset metadata from CSV...")
            self._update_status("preparing", 10)
            dataset_path = os.path.join(self.csv_manager.data_dir, 'dataset.csv') if hasattr(self.csv_manager, 'data_dir') else os.path.join('data', 'dataset.csv')
            df = pd.read_csv(dataset_path)

            # 2. Prepare image paths and labels
            self._log_message("Preparing image paths and labels...")
            self._update_status("preparing", 15)
            image_paths = df['file_path'].tolist()
            labels = df['label'].tolist()
            # Map labels to 0/1
            label_map = {'no_tumor': 0, 'tumor': 1, 'No Brain Tumor': 0, 'Yes Brain Tumor': 1}
            y = np.array([label_map.get(str(l).strip().lower().replace(' ', '_'), 0) for l in labels])
            x = np.array(image_paths)

            # 3. Split into train/val (80/20)
            self._log_message("Splitting data into train and validation sets...")
            self._update_status("preparing", 20)
            idxs = np.arange(len(x))
            np.random.shuffle(idxs)
            split = int(0.8 * len(x))
            train_idxs, val_idxs = idxs[:split], idxs[split:]
            x_train, y_train = x[train_idxs], y[train_idxs]
            x_val, y_val = x[val_idxs], y[val_idxs]

            # 4. Load and preprocess images
            def load_images(paths):
                arr = []
                for p in paths:
                    try:
                        abs_path = p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
                        img = Image.open(abs_path)
                        img = ImageOps.grayscale(img)
                        img = img.resize((150, 150), resample=Image.Resampling.LANCZOS)
                        arr.append(np.asarray(img, dtype=np.float32) / 255.0)
                    except Exception as e:
                        self._log_message(f"Error loading image {p}: {e}")
                        arr.append(np.zeros((150, 150), dtype=np.float32))
                return np.stack(arr)

            self._log_message("Loading and preprocessing training images...")
            self._update_status("preparing", 30)
            X_train = load_images(x_train)
            X_train = X_train[..., np.newaxis]
            self._log_message(f"Loaded {X_train.shape[0]} training images.")

            self._log_message("Loading and preprocessing validation images...")
            self._update_status("preparing", 35)
            X_val = load_images(x_val)
            X_val = X_val[..., np.newaxis]
            self._log_message(f"Loaded {X_val.shape[0]} validation images.")

            # 5. Build model
            self._log_message("Building model architecture...")
            self._update_status("preparing", 40)
            input_shape = (150, 150, 1)
            model = Sequential()
            model.add(Conv2D(16, (3, 3), activation="relu", input_shape=input_shape))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.3))
            model.add(Conv2D(32, (3, 3), activation="relu"))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.3))
            model.add(Conv2D(64, (3, 3), activation="relu"))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.4))
            model.add(Conv2D(128, (3, 3), activation="relu"))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.01)))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # 6. Train model
            self._log_message("Starting model training...")
            self._update_status("training", 50)
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )

            # 7. Evaluate model
            self._log_message("Evaluating model on validation set...")
            self._update_status("training", 80)
            test_loss, test_acc = model.evaluate(X_val, y_val, verbose=1)
            y_pred_probs = model.predict(X_val, verbose=1)
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()
            target_names = ['no_tumor', 'tumor']
            class_report = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)

            # 8. Save model and metadata
            self._log_message("Saving model and metadata...")
            self._update_status("training", 90)
            metadata = {
                "input_shape": model.input_shape,
                "total_parameters": model.count_params(),
                "base_model_type": "CUSTOM",
                "batch_size": batch_size,
                "epochs": epochs,
                "validation_split": 0.2,
                "training_history": {
                    "accuracy": [float(x) for x in history.history['accuracy']],
                    "val_accuracy": [float(x) for x in history.history['val_accuracy']],
                    "loss": [float(x) for x in history.history['loss']],
                    "val_loss": [float(x) for x in history.history['val_loss']]
                },
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "classification_report": class_report
            }
            success = self.model_manager.save_model(model, model_name, version, parent_version, metadata)

            if success:
                self._log_message("Training completed successfully!")
                self._update_status("complete", 100)
                self.training_metrics = {
                    'accuracy': metadata.get('test_accuracy', 0.0),
                    'loss': metadata.get('test_loss', 0.0),
                    'val_accuracy': metadata.get('training_history', {}).get('val_accuracy', [])[-1] if metadata.get('training_history', {}).get('val_accuracy') else 0.0,
                    'val_loss': metadata.get('training_history', {}).get('val_loss', [])[-1] if metadata.get('training_history', {}).get('val_loss') else 0.0
                }
                self._update_training_session(session_id, {
                    'end_time': self.csv_manager.get_current_timestamp(),
                    'status': 'complete',
                    'progress': '100',
                    'accuracy': str(self.training_metrics.get('accuracy', 0.0)),
                    'loss': str(self.training_metrics.get('loss', 0.0)),
                    'logs': json.dumps(self.training_logs)
                })
            else:
                self._log_message("Training failed!")
                self._update_status("failed", self.training_progress)
                self._update_training_session(session_id, {
                    'end_time': self.csv_manager.get_current_timestamp(),
                    'status': 'failed',
                    'logs': json.dumps(self.training_logs)
                })

        except Exception as e:
            self._log_message(f"Error during training: {str(e)}")
            self._update_status("failed", self.training_progress)
            self._update_training_session(session_id, {
                'end_time': self.csv_manager.get_current_timestamp(),
                'status': 'failed',
                'logs': json.dumps(self.training_logs)
            })
    
    def _log_message(self, message: str):
        """Add a log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.training_logs.append(log_entry)
        print(log_entry)  # Also print to console
    
    def _update_status(self, status: str, progress: int):
        """Update training status and progress"""
        self.training_status = status
        self.training_progress = progress
    
    def _update_training_session(self, session_id: str, updates: Dict[str, Any]):
        """Update training session in CSV"""
        self.csv_manager.update_csv_row('training_sessions.csv', session_id, updates)
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'session_id': self.current_training,
            'status': self.training_status,
            'progress': self.training_progress,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'logs': self.training_logs[-10:],  # Last 10 log entries
            'metrics': self.training_metrics
        }
    
    def get_training_logs(self, session_id: Optional[str] = None) -> List[str]:
        """Get training logs for a session"""
        if session_id:
            # Get logs from CSV for specific session
            sessions = self.csv_manager.read_csv('training_sessions.csv')
            for session in sessions:
                if session['id'] == session_id:
                    try:
                        return json.loads(session['logs'])
                    except:
                        return []
            return []
        else:
            # Return current training logs
            return self.training_logs
    
    def get_training_sessions(self) -> List[Dict[str, Any]]:
        """Get all training sessions"""
        sessions = self.csv_manager.read_csv('training_sessions.csv')
        
        # Parse logs and convert string fields to appropriate types
        for session in sessions:
            try:
                session['logs'] = json.loads(session['logs']) if session['logs'] else []
            except:
                session['logs'] = []
            
            # Convert numeric fields
            for field in ['progress', 'epochs', 'dataset_size']:
                if session[field].isdigit():
                    session[field] = int(session[field])
            
            for field in ['accuracy', 'loss']:
                if session[field]:
                    try:
                        session[field] = float(session[field])
                    except:
                        session[field] = 0.0
        
        return sessions
    
    def stop_training(self) -> bool:
        """Stop current training (if possible)"""
        if self.training_status in ["preparing", "training"]:
            self._log_message("Training stop requested...")
            self._update_status("stopped", self.training_progress)
            
            if self.current_training:
                self._update_training_session(self.current_training, {
                    'end_time': self.csv_manager.get_current_timestamp(),
                    'status': 'stopped',
                    'logs': json.dumps(self.training_logs)
                })
            
            return True
        return False
