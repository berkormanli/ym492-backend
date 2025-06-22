from multiprocessing import Lock

from datetime import datetime
import semver
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from model_version_manager import ModelVersion, ModelVersionManager
import numpy as np
from PIL import Image, ImageOps
import cv2
import pickle
from tqdm import tqdm
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.applications import VGG19
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

class BrainTumorModelManager:
    def __init__(self, model_dir: str = "models", cache_dir: str = "cache"):
        self.mutex = Lock()

        self.models: Dict[str, ModelVersionManager] = {}
        self.current_model: Model
        self.current_model_name: str = ""
        self.input_shape = (150, 150, 1)
        self.class_mapping = {0: "No Brain Tumor", 1: "Yes Brain Tumor"}

        # Directory setup
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Load version history from disk if exists
        self.load_version_history()

    def setup_logging(self):
        logging.basicConfig(
            filename='model_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def calculate_model_hash(self, model_or_path) -> str:
        """Calculate SHA-256 hash of model file or model object.

        Args:
            model_or_path: Either a file path to a model or a Keras model object

        Returns:
            SHA-256 hash as a hexadecimal string
        """
        sha256_hash = hashlib.sha256()

        if isinstance(model_or_path, str):
            # It's a file path
            with open(model_or_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
        else:
            # It's a model object - use weights as a proxy for the model
            # Get model weights as a list of numpy arrays
            weights = model_or_path.get_weights()

            # Convert weights to bytes and update hash
            for w in weights:
                sha256_hash.update(w.tobytes())

            # Also include model architecture in the hash
            sha256_hash.update(str(model_or_path.to_json()).encode('utf-8'))

        return sha256_hash.hexdigest()

    def load_version_history(self):
        """Load version history from disk."""
        history_file = self.model_dir / "version_history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)
                for model_name, versions in history.items():
                    if model_name not in self.models:
                        self.models[model_name] = ModelVersionManager()
                    for version_data in versions:
                        # Load the actual model file
                        model_path = self.model_dir / model_name / version_data["version"] / "model.h5"
                        if model_path.exists():
                            model = load_model(str(model_path))
                            version = ModelVersion(
                                version=version_data["version"],
                                model=model,
                                created_at=datetime.fromisoformat(version_data["created_at"]),
                                metadata=version_data["metadata"],
                                parent_version=version_data.get("parent_version")
                            )
                            self.models[model_name].add_version(version)

    def save_version_history(self):
        """Save version history to disk."""
        history = {}
        for model_name, version_manager in self.models.items():
            history[model_name] = version_manager.get_version_history()

        history_file = self.model_dir / "version_history.json"
        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)

    def get_available_models(self) -> list:
        """Return list of available model names."""
        return list(self.models.keys())

    def get_available_models_with_versions(self) -> dict:
        """Return a dictionary of available models with their versions."""
        return {
            model_name: list(self.models[model_name].versions.keys())
            for model_name in self.models
        }

    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess the image for model input."""
        try:
            image = Image.open(image_path)
            image = ImageOps.grayscale(image)
            image = ImageOps.fit(image, (self.input_shape[0], self.input_shape[1]), Image.Resampling.LANCZOS)
            image_array = np.asarray(image).astype(np.float32) / 255.0
            data = image_array.reshape((1, 150, 150, 1))
            return data
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def predict(self, image_path: str) -> Tuple[str, float]:
        """Make prediction using current model version."""
        if not self.current_model_name:
            return "No model selected", 0.0

        version_manager = self.models[self.current_model_name]
        if not version_manager.current_version:
            return "No version selected", 0.0

        #current_version = version_manager.get_version(version_manager.current_version)
        #if not current_version:
        #    return "Version not found", 0.0

        input_img = self.preprocess_image(image_path)
        if input_img is None:
            return "Error processing image", 0.0
        
        with self.mutex:
            result = version_manager.current_version.model.predict(input_img)
            prediction_class = np.argmax(result, axis=1)[0]
            confidence = float(result[0][0])

            if confidence <= 0.5:
                return self.class_mapping[1], confidence
            else:
                return self.class_mapping[0], confidence
        
        return "Mutex lock", 0.0

    def save_model(self, model: Model, model_name: str, version: str,
                  parent_version: Optional[str] = None, metadata: dict = None) -> bool:
        """Save a new version of a model."""
        try:
            # Validate version string
            semver.VersionInfo.parse(version)

            # Calculate model hash
            model_hash = "" #self.calculate_model_hash(model)

            # Check if this version already exists
            if model_name in self.models:
                version_manager = self.models[model_name]
                if version_manager.get_version(version):
                    self.logger.error(f"Version {version} already exists for model {model_name}")
                    return False
            else:
                self.models[model_name] = ModelVersionManager()

            # Merge all possible metadata fields
            full_metadata = {
                "input_shape": model.input_shape,
                "total_parameters": model.count_params(),
                "original_path": str(self.model_dir / model_name / version / "model.h5"),
                "size_mb": os.path.getsize(str(self.model_dir / model_name / version / "model.h5")) / (1024 * 1024) if (self.model_dir / model_name / version / "model.h5").exists() else 0,
            }
            if metadata:
                full_metadata.update(metadata)

            # Create version object
            model_version = ModelVersion(
                version=version,
                model=model,
                created_at=datetime.now(),
                metadata=full_metadata,
                #hash=model_hash,
                parent_version=parent_version
            )

            # Save model file in version directory
            version_dir = self.model_dir / model_name / version
            version_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(version_dir / "model.h5"))
            model.save_weights(str(version_dir / "model.weights.h5"))

            # Update size_mb after saving
            model_version.metadata["size_mb"] = os.path.getsize(str(version_dir / "model.h5")) / (1024 * 1024)

            # Add version to manager
            self.models[model_name].add_version(model_version)

            if not self.current_model_name:
                self.current_model_name = model_name

            # Save updated version history
            self.save_version_history()

            self.logger.info(f"Successfully saved model {model_name} version {version}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving model {model_name} version {version}: {str(e)}")
            return False


    def load_model(self, model_name: str, model_path: str, version: str,
                  parent_version: Optional[str] = None, metadata: dict = None) -> bool:
        """
        Load a new version of a model.

        Args:
            model_name: Name of the model
            model_path: Path to the model file
            version: Semantic version string (e.g., "1.0.0")
            parent_version: Optional parent version string
            metadata: Additional metadata about the model version
        """
        try:
            # Validate version string
            semver.VersionInfo.parse(version)

            # Calculate model hash
            model_hash = "" #self.calculate_model_hash(model_path)

            # Check if this version already exists
            if model_name in self.models:
                print(f"Model {model_name} already exists, checking versions")
                version_manager = self.models[model_name]
                if version_manager.get_version(version):
                    print(f"Version {version} already exists for model {model_name}")
                    self.logger.error(f"Version {version} already exists for model {model_name}")
                    # Load the model
                    model = load_model(model_path)
            else:
                print(f"Model {model_name} not found, creating new version manager")
                self.models[model_name] = ModelVersionManager()
                # Load the model
                model = load_model(model_path)

                # Create version object
                model_version = ModelVersion(
                    version=version,
                    model=model,
                    created_at=datetime.now(),
                    metadata={
                        "input_shape": model.input_shape,
                        "total_parameters": model.count_params(),
                        "original_path": str(model_path),
                        "size_mb": os.path.getsize(model_path) / (1024 * 1024),
                        **(metadata or {})
                    },
                    #hash=model_hash,
                    parent_version=parent_version
                )

                # Add version to manager
                self.models[model_name].add_version(model_version)
            
            self.switch_version(model_name, version)

            if not self.current_model_name:
                self.current_model_name = model_name

            # Save updated version history
            #self.save_version_history()
            print(f"Successfully loaded model {model_name} version {version}")
            self.logger.info(f"Successfully loaded model {model_name} version {version}")
            self.current_model = model
            return True

        except Exception as e:
            print(f"Error loading model {model_name} version {version}: {str(e)}")
            self.logger.error(f"Error loading model {model_name} version {version}: {str(e)}")
            return False

    def switch_version(self, model_name: str, version: str) -> bool:
        """Switch to a specific version of a model."""
        if model_name not in self.models:
            return False

        version_manager = self.models[model_name]
        if version_manager.get_version(version):
            version_manager.current_version = version
            self.current_model_name = model_name
            return True
        return False

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get version history for a specific model."""
        if model_name not in self.models:
            return []
        return self.models[model_name].get_version_history()

    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a specific model."""
        if model_name not in self.models:
            return None
        latest = self.models[model_name].get_latest_version()
        return latest.version if latest else None

    def rollback_version(self, model_name: str) -> bool:
        """Rollback to the previous version of a model."""
        if model_name not in self.models:
            return False

        version_manager = self.models[model_name]
        current = version_manager.get_version(version_manager.current_version)
        if not current or not current.parent_version:
            return False

        return self.switch_version(model_name, current.parent_version)

    def compare_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a model."""
        if model_name not in self.models:
            return {}

        version_manager = self.models[model_name]
        v1 = version_manager.get_version(version1)
        v2 = version_manager.get_version(version2)

        if not v1 or not v2:
            return {}

        return {
            "version1": {
                "version": v1.version,
                "created_at": v1.created_at.isoformat(),
                "metadata": v1.metadata
            },
            "version2": {
                "version": v2.version,
                "created_at": v2.created_at.isoformat(),
                "metadata": v2.metadata
            },
            "changes": {
                "parameters_diff": v2.metadata["total_parameters"] - v1.metadata["total_parameters"],
                "size_diff_mb": v2.metadata["size_mb"] - v1.metadata["size_mb"]
            }
        }

    def prepare_data(self, train_dir: str, test_dir: str, categories: List[str] = None, img_size: int = 150) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and testing data from directories.

        Args:
            train_dir: Directory containing training data organized in category folders
            test_dir: Directory containing testing data organized in category folders
            categories: List of category folder names (default: ["tumor", "no_tumor"])
            img_size: Size to resize images to (default: 150)

        Returns:
            Tuple of (X_train, Y_train, X_test, Y_test)
        """
        if categories is None:
            categories = ["tumor", "no_tumor"]

        self.logger.info(f"Preparing data from {train_dir} and {test_dir}")

        # Create training data
        training_data = []
        for category in categories:
            path = os.path.join(train_dir, category)
            class_num = categories.index(category)
            for img in tqdm(os.listdir(path), desc=f"Loading {category} training data"):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    new_array = cv2.resize(img_array, (img_size, img_size))
                    training_data.append([new_array, class_num])
                except Exception as e:
                    self.logger.error(f"Error loading training image {img}: {str(e)}")

        self.logger.info(f"Training samples: {len(training_data)}")

        X_train = np.array([i[0] for i in training_data]).reshape(-1, img_size, img_size, 1)
        Y_train = np.array([i[1] for i in training_data])

        # Create testing data
        testing_data = []
        for category in categories:
            path = os.path.join(test_dir, category)
            class_num = categories.index(category)
            for img in tqdm(os.listdir(path), desc=f"Loading {category} testing data"):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    new_array = cv2.resize(img_array, (img_size, img_size))
                    testing_data.append([new_array, class_num])
                except Exception as e:
                    self.logger.error(f"Error loading testing image {img}: {str(e)}")

        self.logger.info(f"Testing samples: {len(testing_data)}")

        X_test = np.array([i[0] for i in testing_data]).reshape(-1, img_size, img_size, 1)
        Y_test = np.array([i[1] for i in testing_data])

        # Normalize data
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # Save data to cache
        cache_path = self.cache_dir / "data"
        cache_path.mkdir(exist_ok=True)

        with open(cache_path / "X_train.pickle", "wb") as f:
            pickle.dump(X_train, f)
        with open(cache_path / "Y_train.pickle", "wb") as f:
            pickle.dump(Y_train, f)
        with open(cache_path / "X_test.pickle", "wb") as f:
            pickle.dump(X_test, f)
        with open(cache_path / "Y_test.pickle", "wb") as f:
            pickle.dump(Y_test, f)

        return X_train, Y_train, X_test, Y_test

    def load_cached_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Load data from cache if available."""
        cache_path = self.cache_dir / "data"

        if not cache_path.exists():
            return None

        try:
            with open(cache_path / "X_train.pickle", "rb") as f:
                X_train = pickle.load(f)
            with open(cache_path / "Y_train.pickle", "rb") as f:
                Y_train = pickle.load(f)
            with open(cache_path / "X_test.pickle", "rb") as f:
                X_test = pickle.load(f)
            with open(cache_path / "Y_test.pickle", "rb") as f:
                Y_test = pickle.load(f)

            self.logger.info(f"Loaded data from cache: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
            return X_train, Y_train, X_test, Y_test
        except Exception as e:
            self.logger.error(f"Error loading data from cache: {str(e)}")
            return None

    def create_model(self, img_size: int = 150, base_model_type: str = "SCRATCH") -> Model:
        """Create a new model architecture.

        Args:
            img_size: Input image size (default: 150)
            base_model_type: Base model type (default: "VGG19")

        Returns:
            Compiled Keras model
        """
        self.logger.info(f"Creating new model with {base_model_type} base")

        if base_model_type == "VGG19":
            # Load VGG19 (excluding top classifier layers)
            base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 1))

            # Build new model on top
            model = Sequential()

            # Add VGG19 layers to the model
            for layer in base_model.layers:
                model.add(layer)

            # Freeze layers if necessary
            for layer in model.layers:
                layer.trainable = True  # Can be set to False to freeze weights

            # Add custom classifier layers on top
            model.add(Flatten())
            model.add(Dense(4608, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1152, activation='relu'))
            model.add(Dense(2, activation='softmax'))  # 2 classes: "tumor", "no_tumor"

            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=Adam(learning_rate=1e-6),
                metrics=['accuracy']
            )

            return model
        else:
            from keras.models import Sequential
            from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
            from keras.regularizers import l2

            BATCH_SIZE = 64
            input_shape = (150, 150,1)

            model = Sequential()

            # İlk konvolüsyonel katman
            model.add(Conv2D(16, (3, 3), activation="relu", input_shape=input_shape))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.3))

            # İkinci konvolüsyonel katman
            model.add(Conv2D(32, (3, 3), activation="relu"))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.3))

            # Üçüncü konvolüsyonel katman
            model.add(Conv2D(64, (3, 3), activation="relu"))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.4))

            # Dördüncü konvolüsyonel katman
            model.add(Conv2D(128, (3, 3), activation="relu"))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.5))

            # Özellikleri düzleştir
            model.add(Flatten())

            # Tam bağlantılı katman
            model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.01)))
            model.add(Dropout(0.5))

            # Çıkış katmanı (binary sınıflandırma)
            model.add(Dense(1, activation='sigmoid'))

            # Modeli derle
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Model özeti
            model.summary()

    def train_model(self, model_name: str, version: str, train_dir: str = None, test_dir: str = None,
                   base_model_type: str = "SCRATCH", batch_size: int = 64, epochs: int = 1,
                   validation_split: float = 0.2, parent_version: str = None,
                   use_cached_data: bool = True) -> bool:
        """Train a new model and save it as a new version.

        Args:
            model_name: Name for the model
            version: Version string (e.g., "1.0.0")
            train_dir: Directory containing training data (if None, uses cached data)
            test_dir: Directory containing testing data (if None, uses cached data)
            base_model_type: Base model type (default: "VGG19")
            batch_size: Batch size for training (default: 64)
            epochs: Number of epochs to train (default: 10)
            validation_split: Validation split ratio (default: 0.2)
            parent_version: Optional parent version string
            use_cached_data: Whether to use cached data if available (default: True)

        Returns:
            True if training and saving was successful, False otherwise
        """
        try:
            # Validate version string
            semver.VersionInfo.parse(version)

            # Check if this version already exists
            if model_name in self.models:
                version_manager = self.models[model_name]
                if version_manager.get_version(version):
                    self.logger.error(f"Version {version} already exists for model {model_name}")
                    return False

            # Prepare data
            if use_cached_data:
                cached_data = self.load_cached_data()
                if cached_data is not None:
                    X_train, Y_train, X_test, Y_test = cached_data
                elif train_dir and test_dir:
                    X_train, Y_train, X_test, Y_test = self.prepare_data(train_dir, test_dir)
                else:
                    self.logger.error("No cached data available and no train/test directories provided")
                    return False
            elif train_dir and test_dir:
                X_train, Y_train, X_test, Y_test = self.prepare_data(train_dir, test_dir)
            else:
                self.logger.error("No train/test directories provided and not using cached data")
                return False

            # Create model
            model = self.create_model(img_size=self.input_shape[0], base_model_type=base_model_type)

            # Setup callbacks
            log_dir = self.model_dir / model_name / version / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            tensorboard = TensorBoard(log_dir=str(log_dir), histogram_freq=1, write_graph=True, write_images=False)
            early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
            checkpoint = ModelCheckpoint(
                str(self.model_dir / model_name / version / "best_model.h5"),
                monitor='val_accuracy',
                mode='max',
                verbose=1,
                save_best_only=True
            )

            # Train the model
            self.logger.info(f"Training model {model_name} version {version}")
            history = model.fit(
                X_train, Y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=[tensorboard, early_stopping, checkpoint]
            )

            # Evaluate the model
            scores = model.evaluate(X_test, Y_test, verbose=1)
            self.logger.info(f"Test loss: {scores[0]}, Test accuracy: {scores[1]}")

            # Generate classification report
            y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
            class_report = classification_report(Y_test, y_pred_classes, target_names=list(self.class_mapping.values()), output_dict=True)

            # Create metadata
            metadata = {
                "input_shape": model.input_shape,
                "total_parameters": model.count_params(),
                "base_model_type": base_model_type,
                "batch_size": batch_size,
                "epochs": epochs,
                "validation_split": validation_split,
                "training_history": {
                    "accuracy": [float(x) for x in history.history['accuracy']],
                    "val_accuracy": [float(x) for x in history.history['val_accuracy']],
                    "loss": [float(x) for x in history.history['loss']],
                    "val_loss": [float(x) for x in history.history['val_loss']]
                },
                "test_loss": float(scores[0]),
                "test_accuracy": float(scores[1]),
                "classification_report": class_report
            }

            # Save the model
            return self.save_model(model, model_name, version, parent_version, metadata)

        except Exception as e:
            self.logger.error(f"Error training model {model_name} version {version}: {str(e)}")
            return False

    def evaluate_model(self, model_name: str, version: str = None, test_data: Tuple[np.ndarray, np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate a model version on test data.

        Args:
            model_name: Name of the model
            version: Version to evaluate (if None, uses current version)
            test_data: Optional tuple of (X_test, Y_test) (if None, uses cached test data)

        Returns:
            Dictionary with evaluation results
        """
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}

        version_manager = self.models[model_name]

        if version is None:
            if not version_manager.current_version:
                return {"error": "No current version selected"}
            version = version_manager.current_version

        model_version = version_manager.get_version(version)
        if not model_version:
            return {"error": f"Version {version} not found"}

        # Get test data
        if test_data is None:
            cached_data = self.load_cached_data()
            if cached_data is None:
                return {"error": "No cached test data available"}
            _, _, X_test, Y_test = cached_data
        else:
            X_test, Y_test = test_data

        # Evaluate the model
        scores = model_version.model.evaluate(X_test, Y_test, verbose=1)

        # Generate predictions and classification report
        y_pred = model_version.model.predict(X_test, batch_size=64, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        class_report = classification_report(Y_test, y_pred_classes, target_names=list(self.class_mapping.values()), output_dict=True)
        conf_matrix = confusion_matrix(Y_test, y_pred_classes)

        return {
            "model_name": model_name,
            "version": version,
            "test_loss": float(scores[0]),
            "test_accuracy": float(scores[1]),
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist()
        }