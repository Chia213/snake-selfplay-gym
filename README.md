# Snake Multiplayer Self-Play Gym

A fun multiplayer Snake game where you can play against AI agents or watch them compete! Built with reinforcement learning capabilities.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Choose Your Mode

#### 🎮 **Watch AI Agents Play**
```bash
python demo.py --mode watch
```
- Watch 2 AI snakes compete automatically
- Default: 3 episodes, 12x12 field
- **Fast Mode**: `--fast` flag for minimal delays
- Customize: `--snakes 4 --field_size 20 --episodes 5`

#### 🕹️ **Play Against AI**
```bash
python demo.py --mode play
```
- You control snake 0, AI controls the others
- **Controls**: W (up), S (down), A (left), D (right), Q (quit)
- Customize: `--snakes 3 --field_size 12 --player_snake 1`

#### 🧪 **Test Environment**
```bash
python demo.py --mode test
```
- Quick test to ensure everything works

## 🎯 Game Modes Explained

| Mode | Description | Best For | Performance |
|------|-------------|----------|-------------|
| `watch` | AI vs AI competition | Entertainment, learning | Use `--fast` for speed |
| `play` | Human vs AI | Interactive gaming | Optimized by default |
| `test` | Environment validation | Troubleshooting | Small field for speed |

## 🚀 **Performance Tips**

- **Use `--fast` flag** for watch mode: `python demo.py --mode watch --fast`
- **Smaller field sizes** = faster rendering: `--field_size 10` or `12`
- **Fewer snakes** = better performance: Start with `--snakes 2`
- **Default field size** is now 12x12 (was 15x15) for better performance

## 🎲 Game Features

- **Multiplayer**: 2-4 snakes on the same field
- **Food**: Multiple food items to grow your snake
- **Scoring**: +1 for food, -1 for collisions
- **Field Size**: Configurable (default 15x15)
- **Visual**: Real-time pygame rendering

## 🧠 Training AI (Optional)

If you want to train your own AI agents:

```bash
python simple_train.py
```

This trains a PPO agent for 10,000 steps and saves checkpoints.

## 🧪 **Performance Testing**

Test your setup performance and get optimization recommendations:

```bash
python test_performance.py
```

This will test different configurations and show you the best settings for your system.

## 📁 Project Structure

```
snake-selfplay-gym/
├── demo.py              # 🎮 Main demo script - START HERE!
├── snake_env.py         # Gymnasium environment
├── snake_game.py        # Core game logic
├── simple_train.py      # Basic training script
├── test_performance.py  # 🧪 Performance testing tool
└── requirements.txt     # Dependencies
```

## 🔧 Customization Examples

```bash
# Watch 4 snakes on a large field
python demo.py --mode watch --snakes 4 --field_size 25

# Play as snake 1 against 3 AI opponents
python demo.py --mode play --snakes 4 --player_snake 1

# Quick test with small field
python demo.py --mode test --field_size 10
```

## ❓ Troubleshooting

- **"pygame not found"**: Run `pip install pygame`
- **"gymnasium not found"**: Run `pip install gymnasium`
- **Black screen**: Try `--field_size 15` for smaller fields

## 📜 License

MIT License - Feel free to use, modify, and distribute!

---

**Ready to play?** Just run `python demo.py --mode watch` and enjoy! 🐍


