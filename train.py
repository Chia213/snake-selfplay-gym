#!/usr/bin/env python3
"""
Training script for Snake Multiplayer Self-Play Environment

This script demonstrates how to train agents using various self-play strategies
to overcome training instability in multi-agent reinforcement learning.
"""

import argparse
import os
from selfplay_trainer import SelfPlayTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Snake agents using self-play')
    parser.add_argument('--algorithm', type=str, default='PPO', 
                       choices=['PPO', 'A2C', 'SAC'],
                       help='RL algorithm to use')
    parser.add_argument('--strategy', type=str, default='policy_distribution',
                       choices=['policy_distribution', 'fictitious_selfplay', 
                               'population_based', 'curriculum'],
                       help='Self-play strategy to use')
    parser.add_argument('--snakes', type=int, default=2,
                       help='Number of snakes in the game')
    parser.add_argument('--field_size', type=int, default=20,
                       help='Size of the game field')
    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluation frequency')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models and logs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate for training')
    
    args = parser.parse_args()
    
    print("Snake Multiplayer Self-Play Training")
    print("=" * 50)
    print(f"Algorithm: {args.algorithm}")
    print(f"Self-play Strategy: {args.strategy}")
    print(f"Number of Snakes: {args.snakes}")
    print(f"Field Size: {args.field_size}")
    print(f"Total Timesteps: {args.timesteps}")
    print(f"Save Directory: {args.save_dir}")
    print("=" * 50)
    
    # Create trainer
    trainer = SelfPlayTrainer(
        algorithm=args.algorithm,
        selfplay_strategy=args.strategy,
        num_snakes=args.snakes,
        field_size=args.field_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    # Start training
    try:
        trainer.train(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq
        )
        
        # Plot training progress
        print("\nPlotting training progress...")
        trainer.plot_training_progress()
        
        # Evaluate final model
        print("\nEvaluating final model...")
        results = trainer.evaluate_against_random(num_episodes=100)
        
        print(f"\nTraining completed successfully!")
        print(f"Final Win Rate: {results['win_rate']:.3f}")
        print(f"Final Average Reward: {results['avg_reward']:.3f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer._save_final_model()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
