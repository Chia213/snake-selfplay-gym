#!/usr/bin/env python3
"""
Demo script for Snake Multiplayer Environment

This script demonstrates the environment and allows you to watch AI agents play
or play manually against AI agents.
"""

import numpy as np
import time
from snake_env import SnakeMultiplayerEnv
from stable_baselines3 import PPO
import argparse

def random_agent(obs, legal_actions=None):
    """Random agent that takes random actions"""
    if legal_actions is not None:
        return np.random.choice(legal_actions)
    return np.random.randint(0, 4)

def simple_agent(obs, legal_actions=None):
    """Simple heuristic agent that moves towards food and avoids walls"""
    # This is a placeholder for a simple heuristic
    # In practice, you'd implement food-seeking and collision-avoidance logic
    return np.random.randint(0, 4)

def watch_ai_play(env, num_episodes=3, render_delay=0.1):
    """Watch AI agents play against each other"""
    print(f"Watching AI agents play {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            # Render the environment
            env.render()
            time.sleep(render_delay)
            
            # Generate random actions for all snakes
            actions = np.random.randint(0, 4, size=env.num_snakes)
            
            # Take step
            obs, rewards, terminated, truncated, info = env.step(actions)
            episode_reward += np.mean(rewards)
            step_count += 1
            
            # Print step info
            if step_count % 10 == 0:
                alive_count = sum(info['alive_status'])
                print(f"Step {step_count}: Alive snakes: {alive_count}, "
                      f"Avg reward: {episode_reward/step_count:.3f}")
            
            if terminated or truncated:
                break
        
        # Print episode summary
        final_scores = info['episode_scores']
        print(f"Episode {episode + 1} finished in {step_count} steps")
        print(f"Final scores: {final_scores}")
        print(f"Episode reward: {episode_reward:.3f}")
        print("-" * 50)

def manual_play(env, player_snake_id=0):
    """Allow manual play against AI agents"""
    print(f"Manual play mode - You control snake {player_snake_id}")
    print("Use WASD keys to move:")
    print("W - Up, S - Down, A - Left, D - Right")
    print("Q - Quit")
    
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0
    
    while True:
        # Render the environment
        env.render()
        
        # Get player input
        action = input("Action (W/A/S/D/Q): ").upper()
        
        if action == 'Q':
            break
        
        # Convert input to action
        action_map = {'W': 0, 'S': 2, 'A': 3, 'D': 1}
        if action in action_map:
            player_action = action_map[action]
        else:
            print("Invalid action, using random")
            player_action = np.random.randint(0, 4)
        
        # Generate actions for all snakes
        actions = np.random.randint(0, 4, size=env.num_snakes)
        actions[player_snake_id] = player_action
        
        # Take step
        obs, rewards, terminated, truncated, info = env.step(actions)
        episode_reward += rewards[player_snake_id]
        step_count += 1
        
        # Print info
        alive_count = sum(info['alive_status'])
        print(f"Step {step_count}: Alive snakes: {alive_count}, "
              f"Your reward: {rewards[player_snake_id]:.3f}, "
              f"Total: {episode_reward:.3f}")
        
        if terminated or truncated:
            break
    
    # Print final summary
    final_scores = info['episode_scores']
    print(f"\nGame finished in {step_count} steps")
    print(f"Final scores: {final_scores}")
    print(f"Your final reward: {episode_reward:.3f}")

def test_environment():
    """Test basic environment functionality"""
    print("Testing Snake Multiplayer Environment...")
    
    # Create environment
    env = SnakeMultiplayerEnv(
        num_snakes=2,
        field_size=15,  # Smaller field for testing
        render_mode="human"
    )
    
    print(f"Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Number of snakes: {env.num_snakes}")
    print(f"Field size: {env.field_size}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Reset successful! Observation shape: {obs.shape}")
    
    # Test step
    actions = np.random.randint(0, 4, size=env.num_snakes)
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f"Step successful! Rewards: {rewards}")
    
    # Test rendering
    env.render()
    print("Rendering successful!")
    
    env.close()
    print("Environment test completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Snake Multiplayer Environment Demo')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'watch', 'play'],
                       help='Demo mode')
    parser.add_argument('--snakes', type=int, default=2,
                       help='Number of snakes')
    parser.add_argument('--field_size', type=int, default=15,
                       help='Field size')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to watch')
    parser.add_argument('--player_snake', type=int, default=0,
                       help='Snake ID controlled by player (in play mode)')
    
    args = parser.parse_args()
    
    print("Snake Multiplayer Environment Demo")
    print("=" * 40)
    
    if args.mode == 'test':
        test_environment()
    
    elif args.mode == 'watch':
        env = SnakeMultiplayerEnv(
            num_snakes=args.snakes,
            field_size=args.field_size,
            render_mode="human"
        )
        watch_ai_play(env, num_episodes=args.episodes)
        env.close()
    
    elif args.mode == 'play':
        env = SnakeMultiplayerEnv(
            num_snakes=args.snakes,
            field_size=args.field_size,
            render_mode="human"
        )
        manual_play(env, player_snake_id=args.player_snake)
        env.close()
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
