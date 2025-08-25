# Snake Multiplayer Self-Play Environment

A Gymnasium environment implementing a multiplayer Snake game with self-play reinforcement learning capabilities. This project demonstrates advanced multi-agent reinforcement learning techniques including self-play training and policy distribution approaches.

## Features

- **Multiplayer Snake Game**: Classic Snake gameplay with multiple snakes competing on the same field
- **Gymnasium Environment**: Fully compatible with the Gymnasium API for easy integration with RL algorithms
- **Self-Play Training**: Implements various self-play strategies to overcome training instability
- **Scalable Architecture**: Start with 2 snakes and scale to any number of agents
- **Multiple RL Algorithms**: Support for PPO, A2C, and other Stable-Baselines3 algorithms
- **Visualization Tools**: Real-time game rendering and training progress monitoring

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import gymnasium as gym
from snake_env import SnakeMultiplayerEnv

# Create environment
env = SnakeMultiplayerEnv(num_snakes=2, field_size=20)

# Basic usage
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## Training with Self-Play

```python
from stable_baselines3 import PPO
from selfplay_trainer import SelfPlayTrainer

# Initialize trainer
trainer = SelfPlayTrainer(
    env_id="snake_multiplayer",
    algorithm="PPO",
    selfplay_strategy="policy_distribution"
)

# Start training
trainer.train(total_timesteps=1000000)
```

## Self-Play Strategies

1. **Current vs Past**: Train current policy against a distribution of past policies
2. **Fictitious Self-Play**: Use a mixture of best responses
3. **Population-Based**: Maintain a population of diverse policies
4. **Curriculum Learning**: Gradually increase difficulty

## Environment Details

- **Action Space**: Discrete (4 actions: up, down, left, right)
- **Observation Space**: Multi-dimensional array representing game state
- **Reward Structure**: 
  - +1 for eating food
  - -1 for collision/death
  - Small negative reward for each step to encourage efficiency

## Project Structure

```
snake-selfplay-gym/
├── snake_env.py              # Main environment implementation
├── selfplay_trainer.py       # Self-play training logic
├── snake_game.py             # Core game mechanics
├── agents/                   # Different agent implementations
├── training/                 # Training scripts and utilities
├── evaluation/               # Evaluation and visualization tools
└── experiments/              # Experiment configurations
```

## Research Contributions

This implementation addresses key challenges in multi-agent reinforcement learning:

- **Self-Play Instability**: Implements various techniques to stabilize training
- **Policy Distribution**: Maintains diverse opponent strategies
- **Scalable Multi-Agent**: Efficient handling of variable numbers of agents
- **Competitive Learning**: Emergence of complex strategies through competition

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{snake_multiplayer_selfplay,
  title={Snake Multiplayer Self-Play Environment},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/snake-selfplay-gym}
}
```

## License

MIT License - see LICENSE file for details.


