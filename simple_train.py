#!/usr/bin/env python3
"""
Simple training script for Snake environment

This is a simplified version that focuses on getting basic training working
before implementing the full self-play system.
"""

import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym

# Import our environment
from snake_env import SnakeMultiplayerEnv

def create_simple_env():
    """Create a simple single-agent environment wrapper"""
    class SimpleSnakeEnv(gym.Env):
        def __init__(self):
            super().__init__()
            # Create the underlying snake environment
            self.snake_env = SnakeMultiplayerEnv(
                num_snakes=2,
                field_size=15,
                render_mode=None
            )
            
            # Define action and observation spaces
            self.action_space = self.snake_env.action_space.spaces[0]  # Only first snake
            self.observation_space = self.snake_env.observation_space
            
        def reset(self, seed=None, options=None):
            obs, info = self.snake_env.reset(seed=seed, options=options)
            return obs, info
            
        def step(self, action):
            # Create actions for all snakes (first snake uses our action, others random)
            actions = np.random.randint(0, 4, size=self.snake_env.num_snakes)
            actions[0] = action  # Our agent's action
            
            obs, rewards, terminated, truncated, info = self.snake_env.step(actions)
            
            # Return only the reward for our agent (first snake)
            reward = rewards[0] if len(rewards) > 0 else 0.0
            
            return obs, reward, terminated, truncated, info
            
        def render(self, mode="human"):
            return self.snake_env.render(mode)
            
        def close(self):
            self.snake_env.close()
    
    return SimpleSnakeEnv()

def main():
    print("Simple Snake Training")
    print("=" * 30)
    
    # Create environment
    env = create_simple_env()
    vec_env = DummyVecEnv([lambda: env])
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Create model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="./logs"
    )
    
    # Setup callbacks
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./checkpoints",
        name_prefix="snake_model"
    )
    
    print("Starting training...")
    
    try:
        # Train the model
        model.learn(
            total_timesteps=10000,
            callback=checkpoint_callback
        )
        
        print("Training completed!")
        
        # Save the final model
        model.save("final_snake_model")
        print("Model saved to 'final_snake_model'")
        
        # Test the trained model
        print("\nTesting trained model...")
        test_env = create_simple_env()
        
        obs, info = test_env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(100):  # Test for 100 steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Test completed: {steps} steps, total reward: {total_reward:.3f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
