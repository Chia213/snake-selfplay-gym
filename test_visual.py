#!/usr/bin/env python3
"""
Test script to check visual rendering of the Snake game
"""

import numpy as np
import time
from snake_env import SnakeMultiplayerEnv

def test_visual():
    print("Testing visual rendering...")
    
    # Try to create environment with human rendering
    try:
        env = SnakeMultiplayerEnv(
            num_snakes=2,
            field_size=15,
            render_mode="human"
        )
        print("✅ Environment created with human rendering")
    except Exception as e:
        print(f"❌ Failed to create environment with human rendering: {e}")
        return
    
    # Reset environment
    try:
        obs, info = env.reset()
        print(f"✅ Environment reset successful. Observation shape: {obs.shape}")
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        return
    
    # Try to render
    try:
        print("Attempting to render...")
        env.render()
        print("✅ Render call successful")
        
        # Wait a bit to see if window appears
        print("Waiting 3 seconds for window to appear...")
        time.sleep(3)
        
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        return
    
    # Try a few steps
    try:
        print("Testing a few game steps...")
        for step in range(5):
            actions = np.random.randint(0, 4, size=2)
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            print(f"Step {step + 1}: Actions={actions}, Rewards={rewards}")
            
            # Render each step
            env.render()
            time.sleep(0.5)
            
            if terminated or truncated:
                break
                
        print("✅ Game steps successful")
        
    except Exception as e:
        print(f"❌ Game steps failed: {e}")
        return
    
    # Clean up
    env.close()
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    test_visual()
