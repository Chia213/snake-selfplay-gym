import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from typing import Dict, List, Optional, Tuple, Union, Any
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt

from snake_env import SnakeMultiplayerEnv

class SelfPlayCallback(BaseCallback):
    """Callback for tracking self-play training progress"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        if self.training_env.buf_rews is not None:
            # Calculate episode statistics
            episode_rewards = self.training_env.buf_rews.flatten()
            episode_lengths = self.training_env.buf_lens.flatten()
            
            if len(episode_rewards) > 0:
                self.episode_rewards.extend(episode_rewards)
                self.episode_lengths.extend(episode_lengths)
                
                # Calculate win rate (assuming 2 snakes for now)
                if len(episode_rewards) >= 2:
                    wins = sum(1 for r in episode_rewards if r > 0)
                    win_rate = wins / len(episode_rewards)
                    self.win_rates.append(win_rate)

class PolicyBuffer:
    """Buffer for storing past policies for self-play training"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.policies = []
        self.performance_history = []
        
    def add_policy(self, policy, performance: float):
        """Add a new policy to the buffer"""
        self.policies.append(policy)
        self.performance_history.append(performance)
        
        # Remove oldest policy if buffer is full
        if len(self.policies) > self.max_size:
            self.policies.pop(0)
            self.performance_history.pop(0)
    
    def sample_policy(self, strategy: str = "uniform") -> Any:
        """Sample a policy from the buffer"""
        if not self.policies:
            return None
        
        if strategy == "uniform":
            return np.random.choice(self.policies)
        elif strategy == "performance_weighted":
            # Weight by performance (better policies are sampled more often)
            weights = np.array(self.performance_history)
            weights = weights - weights.min() + 1e-6  # Ensure positive weights
            weights = weights / weights.sum()
            return np.random.choice(self.policies, p=weights)
        elif strategy == "recent":
            # Prefer recent policies
            weights = np.arange(1, len(self.policies) + 1)
            weights = weights / weights.sum()
            return np.random.choice(self.policies, p=weights)
        else:
            return np.random.choice(self.policies)

class SelfPlayTrainer:
    """
    Trainer for self-play reinforcement learning with multiple strategies
    to overcome training instability.
    """
    
    def __init__(
        self,
        env_id: str = "snake_multiplayer-v0",
        algorithm: str = "PPO",
        selfplay_strategy: str = "policy_distribution",
        num_snakes: int = 2,
        field_size: int = 20,
        learning_rate: float = 3e-4,
        buffer_size: int = 100,
        save_dir: str = "models",
        device: str = "auto"
    ):
        self.env_id = env_id
        self.algorithm = algorithm
        self.selfplay_strategy = selfplay_strategy
        self.num_snakes = num_snakes
        self.field_size = field_size
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.save_dir = save_dir
        self.device = device
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "policies"), exist_ok=True)
        
        # Initialize policy buffer
        self.policy_buffer = PolicyBuffer(buffer_size)
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rates': [],
            'policy_performances': []
        }
        
        # Set random seed
        set_random_seed(42)
        
        # Initialize environment and model
        self._setup_environment()
        self._setup_model()
    
    def _setup_environment(self):
        """Setup the training environment"""
        def make_env():
            return SnakeMultiplayerEnv(
                num_snakes=self.num_snakes,
                field_size=self.field_size,
                render_mode=None
            )
        
        self.env = DummyVecEnv([make_env for _ in range(1)])
    
    def _setup_model(self):
        """Setup the RL model"""
        if self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",  # Changed from CnnPolicy to MlpPolicy
                self.env,
                learning_rate=self.learning_rate,
                verbose=1,
                device=self.device,
                tensorboard_log=os.path.join(self.save_dir, "logs")
            )
        elif self.algorithm == "A2C":
            self.model = A2C(
                "MlpPolicy",  # Changed from CnnPolicy to MlpPolicy
                self.env,
                learning_rate=self.learning_rate,
                verbose=1,
                device=self.device,
                tensorboard_log=os.path.join(self.save_dir, "logs")
            )
        elif self.algorithm == "SAC":
            self.model = SAC(
                "MlpPolicy",  # Changed from CnnPolicy to MlpPolicy
                self.env,
                learning_rate=self.learning_rate,
                verbose=1,
                device=self.device,
                tensorboard_log=os.path.join(self.save_dir, "logs")
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _evaluate_policy(self, model, num_episodes: int = 10) -> float:
        """Evaluate a policy against random opponents"""
        total_reward = 0.0
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.env.step(action)
                total_reward += np.mean(rewards)
                done = any(dones)
        
        return total_reward / num_episodes
    
    def _create_opponent_env(self, opponent_policy=None) -> SnakeMultiplayerEnv:
        """Create an environment with a specific opponent policy"""
        env = SnakeMultiplayerEnv(
            num_snakes=self.num_snakes,
            field_size=self.field_size,
            render_mode=None
        )
        
        if opponent_policy is not None:
            # Set opponent actions based on the policy
            env.opponent_policy = opponent_policy
        
        return env
    
    def _train_with_opponent(self, opponent_policy, timesteps: int) -> Dict:
        """Train the current model against a specific opponent"""
        # Create a custom environment wrapper that uses the opponent policy
        class OpponentEnvWrapper(gym.Wrapper):
            def __init__(self, env, opponent_policy):
                super().__init__(env)
                self.opponent_policy = opponent_policy
            
            def step(self, action):
                # For now, assume the first action is for the learning agent
                # and generate opponent actions using the opponent policy
                if hasattr(self.opponent_policy, 'predict'):
                    opponent_action, _ = self.opponent_policy.predict(
                        self.env.get_state(), deterministic=False
                    )
                    # Combine actions (this is a simplified approach)
                    full_action = np.array([action[0]] + [opponent_action[0]])
                else:
                    # Random opponent
                    full_action = np.array([action[0]] + [np.random.randint(0, 4)])
                
                return self.env.step(full_action)
        
        # Create wrapped environment
        wrapped_env = OpponentEnvWrapper(self.env.envs[0], opponent_policy)
        vec_env = DummyVecEnv([lambda: wrapped_env])
        
        # Create a temporary model for this training session
        temp_model = type(self.model)(
            "MlpPolicy",  # Changed from CnnPolicy to MlpPolicy
            vec_env,
            learning_rate=self.learning_rate,
            verbose=0,
            device=self.device
        )
        
        # Copy weights from current model
        temp_model.set_parameters(self.model.get_parameters())
        
        # Train against opponent
        temp_model.learn(total_timesteps=timesteps)
        
        # Evaluate performance
        performance = self._evaluate_policy(temp_model)
        
        return {
            'model': temp_model,
            'performance': performance,
            'opponent': opponent_policy
        }
    
    def train(self, total_timesteps: int = 1000000, eval_freq: int = 10000):
        """Main training loop with self-play"""
        print(f"Starting self-play training with strategy: {self.selfplay_strategy}")
        print(f"Algorithm: {self.algorithm}, Snakes: {self.num_snakes}, Field: {self.field_size}")
        
        # Setup callbacks
        callbacks = [
            SelfPlayCallback(),
            CheckpointCallback(
                save_freq=eval_freq,
                save_path=os.path.join(self.save_dir, "checkpoints"),
                name_prefix="snake_model"
            )
        ]
        
        # Training loop
        timesteps_per_iteration = min(eval_freq, total_timesteps // 10)
        current_timesteps = 0
        
        while current_timesteps < total_timesteps:
            print(f"\nTraining iteration: {current_timesteps}/{total_timesteps}")
            
            if self.selfplay_strategy == "policy_distribution":
                self._train_policy_distribution(timesteps_per_iteration)
            elif self.selfplay_strategy == "fictitious_selfplay":
                self._train_fictitious_selfplay(timesteps_per_iteration)
            elif self.selfplay_strategy == "population_based":
                self._train_population_based(timesteps_per_iteration)
            elif self.selfplay_strategy == "curriculum":
                self._train_curriculum(timesteps_per_iteration)
            else:
                # Standard training
                self.model.learn(total_timesteps=timesteps_per_iteration, callback=callbacks)
            
            current_timesteps += timesteps_per_iteration
            
            # Evaluate current policy
            performance = self._evaluate_policy(self.model)
            self.training_stats['policy_performances'].append(performance)
            
            # Save current policy to buffer
            self.policy_buffer.add_policy(self.model, performance)
            
            # Save training statistics
            self._save_training_stats()
            
            print(f"Current performance: {performance:.3f}")
        
        print("Training completed!")
        self._save_final_model()
    
    def _train_policy_distribution(self, timesteps: int):
        """Train against a distribution of past policies"""
        print("Training against policy distribution...")
        
        # Sample opponent from buffer
        opponent = self.policy_buffer.sample_policy("performance_weighted")
        
        if opponent is not None:
            # Train against this opponent
            result = self._train_with_opponent(opponent, timesteps)
            
            # Update current model with best performing one
            if result['performance'] > self.training_stats['policy_performances'][-1]:
                self.model = result['model']
                print(f"Updated model with performance: {result['performance']:.3f}")
        else:
            # No opponents yet, train normally
            self.model.learn(total_timesteps=timesteps)
    
    def _train_fictitious_selfplay(self, timesteps: int):
        """Train using fictitious self-play (best response)"""
        print("Training with fictitious self-play...")
        
        # Find best opponent from buffer
        if self.policy_buffer.policies:
            best_opponent = max(
                zip(self.policy_buffer.policies, self.policy_buffer.performance_history),
                key=lambda x: x[1]
            )[0]
            
            result = self._train_with_opponent(best_opponent, timesteps)
            
            if result['performance'] > self.training_stats['policy_performances'][-1]:
                self.model = result['model']
        else:
            self.model.learn(total_timesteps=timesteps)
    
    def _train_population_based(self, timesteps: int):
        """Train using population-based approach"""
        print("Training with population-based approach...")
        
        if len(self.policy_buffer.policies) >= 2:
            # Train against multiple opponents
            opponents = np.random.choice(
                self.policy_buffer.policies, 
                size=min(3, len(self.policy_buffer.policies)), 
                replace=False
            )
            
            best_performance = -float('inf')
            best_model = None
            
            for opponent in opponents:
                result = self._train_with_opponent(opponent, timesteps)
                if result['performance'] > best_performance:
                    best_performance = result['performance']
                    best_model = result['model']
            
            if best_model is not None:
                self.model = best_model
        else:
            self.model.learn(total_timesteps=timesteps)
    
    def _train_curriculum(self, timesteps: int):
        """Train using curriculum learning"""
        print("Training with curriculum learning...")
        
        # Gradually increase difficulty
        current_step = len(self.training_stats['policy_performances'])
        
        if current_step < 5:
            # Early stage: train against random opponents
            self.model.learn(total_timesteps=timesteps)
        elif current_step < 15:
            # Middle stage: train against weak opponents
            opponent = self.policy_buffer.sample_policy("uniform")
            if opponent is not None:
                self._train_with_opponent(opponent, timesteps)
            else:
                self.model.learn(total_timesteps=timesteps)
        else:
            # Late stage: train against strong opponents
            self._train_policy_distribution(timesteps)
    
    def _save_training_stats(self):
        """Save training statistics to file"""
        stats_file = os.path.join(self.save_dir, "training_stats.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_stats = {}
        for key, value in self.training_stats.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    serializable_stats[key] = [v.tolist() for v in value]
                elif isinstance(value[0], (np.integer, np.floating)):
                    serializable_stats[key] = [float(v) for v in value]
                else:
                    serializable_stats[key] = value
            else:
                serializable_stats[key] = value
        
        with open(stats_file, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
    
    def _save_final_model(self):
        """Save the final trained model"""
        final_path = os.path.join(self.save_dir, "final_model")
        self.model.save(final_path)
        print(f"Final model saved to: {final_path}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        if not self.training_stats['policy_performances']:
            print("No training data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Policy performance over time
        axes[0, 0].plot(self.training_stats['policy_performances'])
        axes[0, 0].set_title('Policy Performance Over Time')
        axes[0, 0].set_xlabel('Evaluation Step')
        axes[0, 0].set_ylabel('Performance')
        axes[0, 0].grid(True)
        
        # Win rates (if available)
        if self.training_stats['win_rates']:
            axes[0, 1].plot(self.training_stats['win_rates'])
            axes[0, 1].set_title('Win Rate Over Time')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].grid(True)
        
        # Episode rewards (if available)
        if self.training_stats['episode_rewards']:
            # Plot last 1000 rewards
            recent_rewards = self.training_stats['episode_rewards'][-1000:]
            axes[1, 0].plot(recent_rewards)
            axes[1, 0].set_title('Recent Episode Rewards')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].grid(True)
        
        # Episode lengths (if available)
        if self.training_stats['episode_lengths']:
            recent_lengths = self.training_stats['episode_lengths'][-1000:]
            axes[1, 1].plot(recent_lengths)
            axes[1, 1].set_title('Recent Episode Lengths')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Length')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, "training_progress.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved to: {plot_path}")
        
        plt.show()
    
    def evaluate_against_random(self, num_episodes: int = 100) -> Dict:
        """Evaluate the trained model against random opponents"""
        print(f"Evaluating model against random opponents for {num_episodes} episodes...")
        
        wins = 0
        total_rewards = 0.0
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.env.step(action)
                episode_reward += np.mean(rewards)
                episode_length += 1
                done = any(dones)
            
            # Check if our agent won (assuming it's the first snake)
            if episode_reward > 0:
                wins += 1
            
            total_rewards += episode_reward
            episode_lengths.append(episode_length)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}")
        
        results = {
            'win_rate': wins / num_episodes,
            'avg_reward': total_rewards / num_episodes,
            'avg_episode_length': np.mean(episode_lengths),
            'total_episodes': num_episodes
        }
        
        print(f"Evaluation Results:")
        print(f"Win Rate: {results['win_rate']:.3f}")
        print(f"Average Reward: {results['avg_reward']:.3f}")
        print(f"Average Episode Length: {results['avg_episode_length']:.1f}")
        
        return results
