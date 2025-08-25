#!/usr/bin/env python3
"""
Test script to save game state as images
"""

import numpy as np
import matplotlib.pyplot as plt
from snake_env import SnakeMultiplayerEnv

def test_image_save():
    print("Testing image saving...")
    
    # Create environment
    env = SnakeMultiplayerEnv(
        num_snakes=2,
        field_size=15,
        render_mode=None  # No pygame rendering
    )
    
    # Reset environment
    obs, info = env.reset()
    print(f"✅ Environment reset. Observation shape: {obs.shape}")
    
    # Save initial state as image
    try:
        # Get RGB array
        rgb_array = env.render(mode="rgb_array")
        print(f"✅ RGB array created. Shape: {rgb_array.shape}")
        
        # Save as image
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_array)
        plt.title("Snake Game State")
        plt.axis('off')
        
        # Save to file
        plt.savefig("snake_game_state.png", dpi=150, bbox_inches='tight')
        print("✅ Image saved as 'snake_game_state.png'")
        
        plt.close()
        
    except Exception as e:
        print(f"❌ Image saving failed: {e}")
        return
    
    # Test a few steps and save each
    print("Testing game steps and saving images...")
    for step in range(3):
        actions = np.random.randint(0, 4, size=2)
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        print(f"Step {step + 1}: Actions={actions}, Rewards={rewards}")
        
        # Save current state
        try:
            rgb_array = env.render(mode="rgb_array")
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_array)
            plt.title(f"Snake Game - Step {step + 1}")
            plt.axis('off')
            
            plt.savefig(f"snake_step_{step + 1}.png", dpi=150, bbox_inches='tight')
            print(f"✅ Step {step + 1} image saved")
            
            plt.close()
            
        except Exception as e:
            print(f"❌ Step {step + 1} image failed: {e}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("✅ Image test completed!")

if __name__ == "__main__":
    test_image_save()
