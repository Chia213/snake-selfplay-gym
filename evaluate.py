#!/usr/bin/env python3
"""
Evaluation script for Snake Multiplayer Self-Play Environment

This script evaluates trained models and analyzes their behavior patterns.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, A2C, SAC
from snake_env import SnakeMultiplayerEnv
import argparse
import json
from typing import Dict, List, Tuple
import cv2

def load_model(model_path: str, algorithm: str = "PPO"):
    """Load a trained model"""
    if algorithm == "PPO":
        return PPO.load(model_path)
    elif algorithm == "A2C":
        return A2C.load(model_path)
    elif algorithm == "SAC":
        return SAC.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def evaluate_model(model, env, num_episodes: int = 100, render: bool = False) -> Dict:
    """Evaluate a model and collect statistics"""
    print(f"Evaluating model for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    win_rates = []
    food_eaten = []
    survival_times = []
    collision_types = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_food = 0
        episode_collisions = []
        
        while True:
            if render:
                env.render()
            
            # Get model action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, rewards, terminated, truncated, info = env.step(action)
            episode_reward += np.mean(rewards)
            episode_length += 1
            
            # Track food eaten
            for i, snake_info in enumerate(info['snake_info']):
                if snake_info['alive']:
                    current_score = snake_info['score']
                    if current_score > episode_food:
                        episode_food = current_score
            
            # Track collisions
            if terminated:
                for i, snake_info in enumerate(info['snake_info']):
                    if not snake_info['alive']:
                        # Determine collision type (simplified)
                        head = snake_info['head']
                        if (head[0] == 0 or head[0] == env.field_size - 1 or 
                            head[1] == 0 or head[1] == env.field_size - 1):
                            episode_collisions.append('wall')
                        else:
                            episode_collisions.append('snake')
            
            if terminated or truncated:
                break
        
        # Calculate win rate (assuming 2 snakes for now)
        alive_count = sum(1 for snake in info['snake_info'] if snake['alive'])
        if alive_count == 1:
            win_rates.append(1.0)
        else:
            win_rates.append(0.0)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        food_eaten.append(episode_food)
        survival_times.append(episode_length)
        collision_types.extend(episode_collisions)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
    
    # Calculate statistics
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'win_rates': win_rates,
        'food_eaten': food_eaten,
        'survival_times': survival_times,
        'collision_types': collision_types,
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'win_rate': np.mean(win_rates),
        'avg_food': np.mean(food_eaten),
        'avg_survival': np.mean(survival_times),
        'total_episodes': num_episodes
    }
    
    return results

def analyze_behavior(model, env, num_episodes: int = 50) -> Dict:
    """Analyze specific behavior patterns of the model"""
    print("Analyzing behavior patterns...")
    
    behavior_stats = {
        'food_seeking': [],
        'collision_avoidance': [],
        'efficiency': [],
        'aggressive_behavior': []
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_length = 0
        food_positions = []
        snake_positions = []
        
        while True:
            # Get model action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, rewards, terminated, truncated, info = env.step(action)
            episode_length += 1
            
            # Track positions
            for snake_info in info['snake_info']:
                if snake_info['alive']:
                    snake_positions.append(snake_info['head'])
            
            # Track food
            food_positions.extend(env.game.food)
            
            if terminated or truncated:
                break
        
        # Analyze behavior patterns
        if len(snake_positions) > 1 and len(food_positions) > 0:
            # Food seeking efficiency
            snake_head = snake_positions[0] if snake_positions else (0, 0)
            closest_food = min(food_positions, key=lambda f: 
                             abs(f[0] - snake_head[0]) + abs(f[1] - snake_head[1]))
            food_distance = abs(closest_food[0] - snake_head[0]) + abs(closest_food[1] - snake_head[1])
            behavior_stats['food_seeking'].append(food_distance)
            
            # Efficiency (food eaten per step)
            final_score = info['snake_info'][0]['score'] if info['snake_info'] else 0
            efficiency = final_score / max(episode_length, 1)
            behavior_stats['efficiency'].append(efficiency)
    
    # Calculate averages
    for key in behavior_stats:
        if behavior_stats[key]:
            behavior_stats[f'avg_{key}'] = np.mean(behavior_stats[key])
    
    return behavior_stats

def plot_evaluation_results(results: Dict, save_path: str = None):
    """Plot evaluation results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Snake Model Evaluation Results', fontsize=16)
    
    # Episode rewards
    axes[0, 0].hist(results['episode_rewards'], bins=20, alpha=0.7, color='skyblue')
    axes[0, 0].axvline(results['avg_reward'], color='red', linestyle='--', 
                       label=f'Mean: {results["avg_reward"]:.3f}')
    axes[0, 0].set_title('Episode Rewards Distribution')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].hist(results['episode_lengths'], bins=20, alpha=0.7, color='lightgreen')
    axes[0, 1].axvline(results['avg_length'], color='red', linestyle='--',
                       label=f'Mean: {results["avg_length"]:.1f}')
    axes[0, 1].set_title('Episode Lengths Distribution')
    axes[0, 1].set_xlabel('Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Win rates over time
    axes[0, 2].plot(results['win_rates'], alpha=0.7, color='orange')
    axes[0, 2].axhline(results['win_rate'], color='red', linestyle='--',
                       label=f'Overall: {results["win_rate"]:.3f}')
    axes[0, 2].set_title('Win Rate Over Episodes')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Win Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Food eaten
    axes[1, 0].hist(results['food_eaten'], bins=20, alpha=0.7, color='gold')
    axes[1, 0].axvline(results['avg_food'], color='red', linestyle='--',
                       label=f'Mean: {results["avg_food"]:.2f}')
    axes[1, 0].set_title('Food Eaten Distribution')
    axes[1, 0].set_xlabel('Food Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Survival times
    axes[1, 1].hist(results['survival_times'], bins=20, alpha=0.7, color='lightcoral')
    axes[1, 1].axvline(results['avg_survival'], color='red', linestyle='--',
                       label=f'Mean: {results["avg_survival"]:.1f}')
    axes[1, 1].set_title('Survival Times Distribution')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Collision types
    if results['collision_types']:
        collision_counts = {}
        for collision in results['collision_types']:
            collision_counts[collision] = collision_counts.get(collision, 0) + 1
        
        collision_types = list(collision_counts.keys())
        collision_counts_list = list(collision_counts.values())
        
        axes[1, 2].bar(collision_types, collision_counts_list, alpha=0.7, color='plum')
        axes[1, 2].set_title('Collision Types')
        axes[1, 2].set_xlabel('Collision Type')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to: {save_path}")
    
    plt.show()

def save_evaluation_results(results: Dict, behavior_stats: Dict, save_path: str):
    """Save evaluation results to file"""
    # Combine results
    all_results = {
        'evaluation_results': results,
        'behavior_analysis': behavior_stats,
        'summary': {
            'total_episodes': results['total_episodes'],
            'overall_win_rate': results['win_rate'],
            'overall_avg_reward': results['avg_reward'],
            'overall_avg_length': results['avg_length'],
            'overall_avg_food': results['avg_food']
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Evaluation results saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Snake models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'A2C', 'SAC'],
                       help='Algorithm used for training')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                       help='Render evaluation episodes')
    parser.add_argument('--save_results', type=str, default='evaluation_results.json',
                       help='File to save evaluation results')
    parser.add_argument('--save_plots', type=str, default='evaluation_plots.png',
                       help='File to save evaluation plots')
    parser.add_argument('--snakes', type=int, default=2,
                       help='Number of snakes in environment')
    parser.add_argument('--field_size', type=int, default=20,
                       help='Field size for environment')
    
    args = parser.parse_args()
    
    print("Snake Model Evaluation")
    print("=" * 40)
    print(f"Model: {args.model_path}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Snakes: {args.snakes}")
    print(f"Field size: {args.field_size}")
    print("=" * 40)
    
    # Load model
    try:
        model = load_model(args.model_path, args.algorithm)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Create environment
    env = SnakeMultiplayerEnv(
        num_snakes=args.snakes,
        field_size=args.field_size,
        render_mode="human" if args.render else None
    )
    
    # Evaluate model
    results = evaluate_model(model, env, args.num_episodes, args.render)
    
    # Analyze behavior
    behavior_stats = analyze_behavior(model, env, min(50, args.num_episodes))
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total Episodes: {results['total_episodes']}")
    print(f"Win Rate: {results['win_rate']:.3f}")
    print(f"Average Reward: {results['avg_reward']:.3f}")
    print(f"Average Episode Length: {results['avg_length']:.1f}")
    print(f"Average Food Eaten: {results['avg_food']:.2f}")
    
    if 'avg_food_seeking' in behavior_stats:
        print(f"Average Food Distance: {behavior_stats['avg_food_seeking']:.2f}")
    if 'avg_efficiency' in behavior_stats:
        print(f"Average Efficiency: {behavior_stats['avg_efficiency']:.3f}")
    
    # Plot results
    plot_evaluation_results(results, args.save_plots)
    
    # Save results
    save_evaluation_results(results, behavior_stats, args.save_results)
    
    env.close()
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
