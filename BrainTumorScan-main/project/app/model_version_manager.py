from dataclasses import dataclass
import semver
from typing import Dict, List, Optional, Any
from keras.models import Model
from datetime import datetime
import json

@dataclass
class ModelVersion:
    version: str
    model: Model
    created_at: datetime
    metadata: Dict[str, Any]
    #hash: str
    parent_version: Optional[str] = None
    
class ModelVersionManager:
    def __init__(self, csv_manager=None, model_manager=None):
        self.versions: Dict[str, ModelVersion] = {}
        self.current_version: Optional[str] = None  # store version string
        self.csv_manager = csv_manager
        self.model_manager = model_manager
    
    def add_version(self, version: ModelVersion):
        self.versions[version.version] = version
        if not self.current_version:
            self.current_version = version.version
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        return self.versions.get(version)
    
    def get_latest_version(self) -> Optional[ModelVersion]:
        if not self.versions:
            return None
        latest = max(self.versions.keys(), key=lambda v: semver.VersionInfo.parse(v))
        return self.versions[latest]

    def get_best_accuracy_version(self) -> Optional[ModelVersion]:
        if not self.versions:
            return None
        best = max(self.versions.values(), key=lambda v: v.metadata.get("accuracy", 0))
        return best

    def get_best_loss_version(self) -> Optional[ModelVersion]:
        if not self.versions:
            return None
        best = min(self.versions.values(), key=lambda v: v.metadata.get("loss", float('inf')))
        return best 
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        return [
            {
                "version": v.version,
                "created_at": v.created_at.isoformat(),
                "parent_version": v.parent_version,
                "metadata": v.metadata
            }
            for v in self.versions.values()
        ]

    def get_available_models(self) -> Dict[str, Any]:
        """Get all available models with their versions"""
        models_info = {}
        if not self.model_manager:
            return models_info
        for model_name, version_manager in self.model_manager.models.items():
            versions = []
            for version_str, model_version in version_manager.versions.items():
                version_info = {
                    'version': version_str,
                    'created_at': model_version.created_at.isoformat(),
                    'metadata': model_version.metadata,
                    'parent_version': model_version.parent_version,
                    'is_current': version_manager.current_version == version_str
                }
                versions.append(version_info)
            models_info[model_name] = {
                'versions': versions,
                'current_version': version_manager.current_version,
                'total_versions': len(versions)
            }
        return models_info

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get versions for a specific model"""
        if not self.model_manager or model_name not in self.model_manager.models:
            return []
        version_manager = self.model_manager.models[model_name]
        versions = []
        for version_str, model_version in version_manager.versions.items():
            version_info = {
                'version': version_str,
                'created_at': model_version.created_at.isoformat(),
                'metadata': model_version.metadata,
                'parent_version': model_version.parent_version,
                'is_current': version_manager.current_version == version_str
            }
            versions.append(version_info)
        return versions

    def get_model_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a model"""
        if not self.model_manager or model_name not in self.model_manager.models:
            return {}
        version_manager = self.model_manager.models[model_name]
        if not version_manager.current_version:
            return {}
        current_version = version_manager.versions[version_manager.current_version]
        metadata = current_version.metadata
        return {
            'current_version': version_manager.current_version,
            'accuracy': metadata.get('test_accuracy', 0.0),
            'loss': metadata.get('test_loss', 0.0),
            'total_parameters': metadata.get('total_parameters', 0),
            'input_shape': metadata.get('input_shape'),
            'training_history': metadata.get('training_history', {}),
            'classification_report': metadata.get('classification_report', {})
        }

    def switch_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Switch to a different model or version"""
        try:
            if not self.model_manager:
                return False
            if version:
                # Switch to specific version
                if model_name in self.model_manager.models:
                    version_manager = self.model_manager.models[model_name]
                    if version in version_manager.versions:
                        version_manager.current_version = version
                        self.model_manager.current_model_name = model_name
                        return True
                return False
            else:
                # Switch to different model (latest version)
                return self.model_manager.switch_model(model_name)
        except Exception:
            return False

    def delete_model_version(self, model_name: str, version: str) -> bool:
        """Delete a specific model version"""
        try:
            if not self.model_manager or model_name not in self.model_manager.models:
                return False
            version_manager = self.model_manager.models[model_name]
            if version not in version_manager.versions:
                return False
            # Don't allow deleting the current version
            if version_manager.current_version == version:
                return False
            # Delete the version
            del version_manager.versions[version]
            # Delete model files if they exist
            model_dir = self.model_manager.model_dir / model_name / version
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
            return True
        except Exception:
            return False