"""
Configuration loader for consistent training parameters
"""
import os
import yaml
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage training configurations for fair comparison"""
    
    def __init__(self, config_path: str = "configs/training_configs.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration if file is missing"""
        return {
            'cityscapes': {
                'epochs': 100, 'batch_size': 8, 'lr': 0.01, 'weight_decay': 1e-4,
                'scheduler': 'poly', 'num_workers': 4, 'input_size': [512, 1024]
            },
            'camvid': {
                'epochs': 200, 'batch_size': 16, 'lr': 0.01, 'weight_decay': 1e-4,
                'scheduler': 'poly', 'num_workers': 4, 'input_size': [360, 480]
            }
        }
    
    def get_training_config(self, dataset: str, model: str = None, method: str = None) -> Dict[str, Any]:
        """
        Get training configuration for specific dataset/model/method combination
        
        Args:
            dataset: Dataset name ('cityscapes' or 'camvid')
            model: Model name (optional, for model-specific settings)
            method: Pruning method (optional, for method-specific settings)
        
        Returns:
            Dictionary of training parameters
        """
        if dataset not in self.config:
            raise ValueError(f"Dataset {dataset} not found in config")
        
        # Start with base dataset config
        config = self.config[dataset].copy()
        
        # Apply model-specific settings if available
        if model and 'models' in self.config[dataset] and model in self.config[dataset]['models']:
            model_config = self.config[dataset]['models'][model]
            config.update(model_config)
        
        # Apply method-specific settings if available
        if method and 'pruning_methods' in self.config and method in self.config['pruning_methods']:
            method_config = self.config['pruning_methods'][method]
            config.update(method_config)
            
            # Handle extra epochs for certain methods
            if 'extra_epochs' in method_config:
                config['epochs'] = config.get('epochs', 100) + method_config['extra_epochs']
        
        return config
    
    def get_finetuning_config(self, dataset: str) -> Dict[str, Any]:
        """Get finetuning configuration for dataset"""
        if 'finetuning' not in self.config or dataset not in self.config['finetuning']:
            # Default finetuning config
            base_config = self.get_training_config(dataset)
            return {
                'epochs': max(50, base_config.get('epochs', 100) // 2),
                'lr': base_config.get('lr', 0.01) * 0.5,
                'batch_size': base_config.get('batch_size', 8),
                'scheduler': base_config.get('scheduler', 'poly'),
                'weight_decay': base_config.get('weight_decay', 1e-4)
            }
        
        return self.config['finetuning'][dataset].copy()
    
    def get_evaluation_config(self, dataset: str) -> Dict[str, Any]:
        """Get evaluation configuration"""
        eval_config = self.config.get('evaluation', {
            'batch_size': 4, 'num_workers': 4, 'oracle_samples': 32
        }).copy()
        
        # Add dataset-specific input size
        if 'input_sizes' in self.config.get('evaluation', {}):
            input_sizes = self.config['evaluation']['input_sizes']
            if dataset in input_sizes:
                eval_config['input_size'] = input_sizes[dataset]
        
        return eval_config
    
    def get_benchmark_config(self, test_type: str = 'basic') -> Dict[str, Any]:
        """
        Get benchmark configuration
        
        Args:
            test_type: Type of benchmark ('basic', 'advanced', 'full', 'quick')
        """
        benchmark_config = self.config.get('benchmark', {}).copy()
        
        # Select appropriate ratios and methods
        if test_type == 'quick':
            ratios = benchmark_config.get('quick_ratios', [0.2, 0.3, 0.5])
            methods = benchmark_config.get('method_groups', {}).get('basic', 
                     ['taylor', 'fpgm', 'l1', 'l2', 'random'])
        elif test_type == 'full':
            ratios = benchmark_config.get('full_ratios', 
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            methods = benchmark_config.get('method_groups', {}).get('full',
                     ['taylor', 'slimming', 'fpgm', 'l1', 'l2', 'random', 'dmcp', 'fgp', 'sirfp'])
        elif test_type == 'advanced':
            ratios = benchmark_config.get('ratios', [0.1, 0.2, 0.3, 0.4, 0.5])
            methods = benchmark_config.get('method_groups', {}).get('advanced',
                     ['taylor', 'slimming', 'fpgm', 'fgp', 'sirfp'])
        else:  # basic
            ratios = benchmark_config.get('ratios', [0.1, 0.2, 0.3, 0.4, 0.5])
            methods = benchmark_config.get('method_groups', {}).get('basic',
                     ['taylor', 'fpgm', 'l1', 'l2', 'random'])
        
        return {
            'ratios': ratios,
            'methods': methods
        }
    
    def get_hardware_requirements(self, dataset: str) -> Dict[str, str]:
        """Get hardware requirements for dataset"""
        if 'hardware' not in self.config or dataset not in self.config['hardware']:
            return {
                'min_gpu_memory': '8GB',
                'recommended_gpu_memory': '12GB', 
                'min_system_memory': '16GB'
            }
        
        return self.config['hardware'][dataset].copy()
    
    def print_config_summary(self, dataset: str, model: str = None, method: str = None):
        """Print configuration summary for verification"""
        config = self.get_training_config(dataset, model, method)
        
        print(f"\n=== Configuration Summary ===")
        print(f"Dataset: {dataset}")
        if model:
            print(f"Model: {model}")
        if method:
            print(f"Method: {method}")
        print(f"Epochs: {config.get('epochs')}")
        print(f"Batch Size: {config.get('batch_size')}")
        print(f"Learning Rate: {config.get('lr')}")
        print(f"Weight Decay: {config.get('weight_decay')}")
        print(f"Scheduler: {config.get('scheduler')}")
        print(f"Input Size: {config.get('input_size')}")
        
        if method == 'slimming' and 'slimming_lambda' in config:
            print(f"Slimming Lambda: {config.get('slimming_lambda')}")
        
        print("=" * 30)


# Global config loader instance
config_loader = ConfigLoader()


def get_training_args(dataset: str, model: str = None, method: str = None) -> Dict[str, Any]:
    """Convenience function to get training arguments"""
    return config_loader.get_training_config(dataset, model, method)


def get_finetuning_args(dataset: str) -> Dict[str, Any]:
    """Convenience function to get finetuning arguments"""
    return config_loader.get_finetuning_config(dataset)


def get_evaluation_args(dataset: str) -> Dict[str, Any]:
    """Convenience function to get evaluation arguments"""
    return config_loader.get_evaluation_config(dataset)


def get_benchmark_args(test_type: str = 'basic') -> Dict[str, Any]:
    """Convenience function to get benchmark arguments"""
    return config_loader.get_benchmark_config(test_type)
