"""
Configuration Loader for MedMNIST Training Pipeline

This module provides utilities for loading and managing YAML configuration files
with support for:
- Importing and merging multiple configuration files
- Environment variable substitution
- Configuration validation
- Default value handling
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy
import warnings


class ConfigLoader:
    """
    Configuration loader that handles YAML files with import support.
    
    Supports importing base configurations and merging them with
    dataset-specific or experiment-specific configurations.
    """
    
    def __init__(self, config_dir: str = "./config"):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.loaded_configs = {}
        self.merged_config = {}
    
    def load_config(self, config_file: str, resolve_imports: bool = True) -> Dict[str, Any]:
        """
        Load a configuration file with optional import resolution.
        
        Args:
            config_file: Name of the configuration file (e.g., 'organmnist3d_config.yaml')
            resolve_imports: Whether to resolve import statements
            
        Returns:
            Merged configuration dictionary
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load the main configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if resolve_imports:
            config = self._resolve_imports(config)
        
        # Store the loaded configuration
        self.loaded_configs[config_file] = config
        self.merged_config = copy.deepcopy(config)
        
        return self.merged_config
    
    def _resolve_imports(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve import statements in the configuration.
        
        Args:
            config: Configuration dictionary with potential imports
            
        Returns:
            Configuration with imports resolved
        """
        if 'imports' not in config:
            return config
        
        # Start with an empty config
        merged_config = {}
        
        # Process imports in order (base configs first)
        for import_file in config['imports']:
            import_path = self.config_dir / import_file
            
            if not import_path.exists():
                warnings.warn(f"Import file not found: {import_path}")
                continue
            
            # Load and merge imported config
            with open(import_path, 'r') as f:
                imported_config = yaml.safe_load(f)
            
            # Recursively resolve imports in the imported config
            imported_config = self._resolve_imports(imported_config)
            
            # Merge with current config
            merged_config = self._deep_merge(merged_config, imported_config)
        
        # Remove imports from the main config
        main_config = copy.deepcopy(config)
        del main_config['imports']
        
        # Merge main config (overrides imported configs)
        merged_config = self._deep_merge(merged_config, main_config)
        
        return merged_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'data.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.merged_config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_config(self, key: str, value: Any):
        """
        Set a configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'data.batch_size')
            value: Value to set
        """
        keys = key.split('.')
        config = self.merged_config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with multiple key-value pairs.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            self.set_config(key, value)
    
    def validate_config(self) -> bool:
        """
        Validate the loaded configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_keys = [
            'data.dataset_name',
            'model.num_classes',
            'training.max_epochs',
            'hardware.gpus'
        ]
        
        for key in required_keys:
            if self.get_config(key) is None:
                warnings.warn(f"Missing required configuration key: {key}")
                return False
        
        return True
    
    def print_config(self, indent: int = 0):
        """
        Print the current configuration in a readable format.
        
        Args:
            indent: Indentation level for nested structures
        """
        def _print_dict(d, level=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    print("  " * level + f"{key}:")
                    _print_dict(value, level + 1)
                else:
                    print("  " * level + f"{key}: {value}")
        
        print("Current Configuration:")
        print("=" * 50)
        _print_dict(self.merged_config)
    
    def save_config(self, output_path: str):
        """
        Save the current configuration to a file.
        
        Args:
            output_path: Path to save the configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.merged_config, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to: {output_path}")
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.get_config('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.get_config('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self.get_config('training', {})
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration section."""
        return self.get_config('hardware', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self.get_config('logging', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration section."""
        return self.get_config('evaluation', {})
    
    def get_test_mode_config(self) -> Dict[str, Any]:
        """Get test mode configuration section."""
        return self.get_config('test_mode', {})
    
    def is_test_mode(self) -> bool:
        """Check if test mode is enabled."""
        return self.get_config('test_mode.enabled', False)


def load_config(config_file: str, config_dir: str = "./config") -> ConfigLoader:
    """
    Convenience function to load a configuration file.
    
    Args:
        config_file: Name of the configuration file
        config_dir: Directory containing configuration files
        
    Returns:
        Configured ConfigLoader instance
    """
    loader = ConfigLoader(config_dir)
    loader.load_config(config_file)
    return loader


def create_test_config() -> ConfigLoader:
    """
    Create a test configuration for quick validation.
    
    Returns:
        ConfigLoader with test configuration
    """
    return load_config('test_config.yaml')


def create_organmnist3d_config() -> ConfigLoader:
    """
    Create OrganMNIST3D configuration.
    
    Returns:
        ConfigLoader with OrganMNIST3D configuration
    """
    return load_config('organmnist3d_config.yaml')


if __name__ == "__main__":
    # Test the configuration loader
    print("Testing Configuration Loader...")
    
    try:
        # Test loading test configuration
        config = create_test_config()
        print("✅ Test configuration loaded successfully")
        
        # Test configuration access
        batch_size = config.get_config('data.batch_size')
        print(f"✅ Batch size: {batch_size}")
        
        # Test configuration validation
        is_valid = config.validate_config()
        print(f"✅ Configuration valid: {is_valid}")
        
        # Print configuration
        config.print_config()
        
    except Exception as e:
        print(f"❌ Error testing configuration loader: {e}")
        import traceback
        traceback.print_exc()

