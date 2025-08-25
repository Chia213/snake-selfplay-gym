"""
Configuration file for Snake Multiplayer Self-Play Environment

This file contains various experiment configurations and hyperparameters
for training and evaluation.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    
    # Environment settings
    num_snakes: int = 2
    field_size: int = 20
    food_count: int = 3
    max_steps: int = 800  # field_size * field_size * 2
    
    # Training parameters
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    learning_rate: float = 3e-4
    buffer_size: int = 100
    
    # Algorithm settings
    algorithm: str = "PPO"
    selfplay_strategy: str = "policy_distribution"
    
    # Model settings
    policy_type: str = "CnnPolicy"
    device: str = "auto"
    
    # Logging and saving
    save_dir: str = "models"
    log_interval: int = 100
    save_interval: int = 10000

@dataclass
class EnvironmentConfig:
    """Configuration for environment parameters"""
    
    # Game mechanics
    reward_scale: float = 1.0
    food_reward: float = 1.0
    death_penalty: float = -1.0
    step_penalty: float = -0.01
    survival_bonus: float = 5.0
    
    # Collision detection
    wall_collision: bool = True
    self_collision: bool = True
    snake_collision: bool = True
    
    # Food spawning
    food_spawn_rate: float = 1.0
    max_food: int = 5
    
    # Snake properties
    initial_length: int = 1
    growth_rate: int = 1

@dataclass
class SelfPlayConfig:
    """Configuration for self-play strategies"""
    
    # Policy distribution
    distribution_strategy: str = "performance_weighted"  # uniform, performance_weighted, recent
    
    # Fictitious self-play
    best_response_threshold: float = 0.1
    
    # Population-based
    population_size: int = 10
    tournament_size: int = 3
    
    # Curriculum learning
    curriculum_stages: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.curriculum_stages is None:
            self.curriculum_stages = [
                {
                    'name': 'random_opponents',
                    'timesteps': 50000,
                    'description': 'Train against random opponents'
                },
                {
                    'name': 'weak_opponents',
                    'timesteps': 100000,
                    'description': 'Train against weak opponents'
                },
                {
                    'name': 'strong_opponents',
                    'timesteps': 200000,
                    'description': 'Train against strong opponents'
                }
            ]

@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    
    num_episodes: int = 100
    render_eval: bool = False
    save_videos: bool = False
    video_dir: str = "videos"
    
    # Evaluation metrics
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                'win_rate',
                'avg_reward',
                'avg_episode_length',
                'survival_rate',
                'food_efficiency'
            ]

@dataclass
class ExperimentConfig:
    """Main configuration for experiments"""
    
    name: str = "snake_selfplay_experiment"
    description: str = "Multiplayer Snake self-play training experiment"
    
    # Configurations
    training: TrainingConfig = None
    environment: EnvironmentConfig = None
    selfplay: SelfPlayConfig = None
    evaluation: EvaluationConfig = None
    
    # Experiment tracking
    seed: int = 42
    log_level: str = "INFO"
    tensorboard_log: bool = True
    
    def __post_init__(self):
        if self.training is None:
            self.training = TrainingConfig()
        if self.environment is None:
            self.environment = EnvironmentConfig()
        if self.selfplay is None:
            self.selfplay = SelfPlayConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()

# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    "basic_2snakes": ExperimentConfig(
        name="basic_2snakes",
        description="Basic 2-snake self-play training",
        training=TrainingConfig(
            num_snakes=2,
            field_size=15,
            total_timesteps=500000
        )
    ),
    
    "advanced_4snakes": ExperimentConfig(
        name="advanced_4snakes",
        description="Advanced 4-snake self-play training",
        training=TrainingConfig(
            num_snakes=4,
            field_size=25,
            total_timesteps=2000000
        ),
        selfplay=SelfPlayConfig(
            selfplay_strategy="population_based",
            population_size=20
        )
    ),
    
    "curriculum_learning": ExperimentConfig(
        name="curriculum_learning",
        description="Curriculum learning experiment",
        training=TrainingConfig(
            selfplay_strategy="curriculum",
            total_timesteps=1500000
        )
    ),
    
    "policy_distribution": ExperimentConfig(
        name="policy_distribution",
        description="Policy distribution self-play",
        training=TrainingConfig(
            selfplay_strategy="policy_distribution",
            total_timesteps=1000000
        ),
        selfplay=SelfPlayConfig(
            distribution_strategy="performance_weighted"
        )
    )
}

def load_config(config_name: str) -> ExperimentConfig:
    """Load a predefined experiment configuration"""
    if config_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(EXPERIMENT_CONFIGS.keys())}")
    
    return EXPERIMENT_CONFIGS[config_name]

def save_config(config: ExperimentConfig, filepath: str):
    """Save configuration to file"""
    import json
    
    # Convert dataclass to dict
    def dataclass_to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: dataclass_to_dict(v) for k, v in obj.items()}
        else:
            return obj
    
    config_dict = dataclass_to_dict(config)
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config_from_file(filepath: str) -> ExperimentConfig:
    """Load configuration from file"""
    import json
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    # Reconstruct dataclass objects
    def dict_to_dataclass(data, cls):
        if hasattr(cls, '__dataclass_fields__'):
            field_values = {}
            for field_name, field_type in cls.__dataclass_fields__.items():
                if field_name in data:
                    field_values[field_name] = dict_to_dataclass(data[field_name], field_type)
                else:
                    field_values[field_name] = field_type()
            return cls(**field_values)
        else:
            return data
    
    # This is a simplified approach - in practice you'd want more robust reconstruction
    return dict_to_dataclass(config_dict, ExperimentConfig)

if __name__ == "__main__":
    # Example usage
    config = load_config("basic_2snakes")
    print(f"Loaded config: {config.name}")
    print(f"Description: {config.description}")
    print(f"Snakes: {config.training.num_snakes}")
    print(f"Field size: {config.training.field_size}")
    
    # Save config
    save_config(config, "experiment_config.json")
    print("Config saved to experiment_config.json")
