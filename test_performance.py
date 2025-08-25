#!/usr/bin/env python3
"""
Performance test script for Snake Multiplayer Environment

This script tests the loading time and performance of different configurations.
"""

import time
import numpy as np
from snake_env import SnakeMultiplayerEnv

def test_loading_time(field_size, num_snakes, fast_render=False):
    """Test how long it takes to create and initialize the environment"""
    print(f"Testing {field_size}x{field_size} field with {num_snakes} snakes...")
    
    start_time = time.time()
    
    # Create environment
    env = SnakeMultiplayerEnv(
        num_snakes=num_snakes,
        field_size=field_size,
        render_mode="human",
        fast_render=fast_render
    )
    
    creation_time = time.time() - start_time
    print(f"  Environment creation: {creation_time:.3f}s")
    
    # Test reset time
    start_time = time.time()
    obs, info = env.reset()
    reset_time = time.time() - start_time
    print(f"  Reset time: {reset_time:.3f}s")
    
    # Test step time
    start_time = time.time()
    actions = np.random.randint(0, 4, size=num_snakes)
    obs, rewards, terminated, truncated, info = env.step(actions)
    step_time = time.time() - start_time
    print(f"  Step time: {step_time:.3f}s")
    
    # Test render time
    start_time = time.time()
    env.render()
    render_time = time.time() - start_time
    print(f"  Render time: {render_time:.3f}s")
    
    total_time = creation_time + reset_time + step_time + render_time
    print(f"  Total test time: {total_time:.3f}s")
    
    env.close()
    return total_time

def main():
    print("Snake Multiplayer Environment - Performance Test")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        (10, 2, False),   # Small field, 2 snakes, normal render
        (12, 2, False),   # Medium field, 2 snakes, normal render
        (12, 2, True),    # Medium field, 2 snakes, fast render
        (15, 2, False),   # Large field, 2 snakes, normal render
        (15, 2, True),    # Large field, 2 snakes, fast render
        (20, 2, True),    # Very large field, 2 snakes, fast render
        (12, 4, True),    # Medium field, 4 snakes, fast render
    ]
    
    results = []
    
    for field_size, num_snakes, fast_render in configs:
        print(f"\n{'='*20}")
        total_time = test_loading_time(field_size, num_snakes, fast_render)
        results.append((field_size, num_snakes, fast_render, total_time))
    
    # Summary
    print(f"\n{'='*50}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    
    for field_size, num_snakes, fast_render, total_time in results:
        render_mode = "FAST" if fast_render else "NORMAL"
        print(f"{field_size:2d}x{field_size:2d} field, {num_snakes} snakes, {render_mode:5s}: {total_time:.3f}s")
    
    # Recommendations
    print(f"\n{'='*50}")
    print("RECOMMENDATIONS")
    print(f"{'='*50}")
    print("✅ For best performance: Use --field_size 10 or 12 with --fast flag")
    print("✅ For casual play: --field_size 12 with 2-3 snakes")
    print("✅ For testing: --field_size 10 with 2 snakes")
    print("❌ Avoid: --field_size 20+ without --fast flag")

if __name__ == "__main__":
    main()
