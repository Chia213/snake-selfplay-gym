import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import pygame
import cv2

from snake_game import SnakeGame, Direction

class SnakeMultiplayerEnv(gym.Env):
    """
    Multiplayer Snake environment for Gymnasium.
    
    This environment implements a multiplayer Snake game where multiple snakes
    compete on the same field. The game ends when all snakes die or only one remains.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "state"],
        "render_fps": 10,
    }
    
    def __init__(
        self,
        num_snakes: int = 2,
        field_size: int = 20,
        food_count: int = 3,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        reward_scale: float = 1.0
    ):
        super().__init__()
        
        self.num_snakes = num_snakes
        self.field_size = field_size
        self.food_count = food_count
        self.render_mode = render_mode
        self.reward_scale = reward_scale
        
        # Initialize the game
        self.game = SnakeGame(field_size, num_snakes, food_count)
        
        # Set max steps
        if max_steps is None:
            self.max_steps = field_size * field_size * 2
        else:
            self.max_steps = max_steps
        
        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([4] * num_snakes)  # 4 actions per snake
        
        # Observation space: field_size x field_size x 4 channels
        # Channels: snake bodies, snake heads, food, walls
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(field_size, field_size, 4), 
            dtype=np.float32
        )
        
        # Initialize pygame for rendering if needed
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((field_size * 20, field_size * 20))
            pygame.display.set_caption("Snake Multiplayer")
            self.clock = pygame.time.Clock()
        
        # Track episode statistics
        self.episode_steps = 0
        self.episode_rewards = [0.0] * num_snakes
        self.episode_scores = [0] * num_snakes
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset the game
        self.game.reset()
        
        # Reset episode tracking
        self.episode_steps = 0
        self.episode_rewards = [0.0] * self.num_snakes
        self.episode_scores = [0] * self.num_snakes
        
        # Get initial observation
        observation = self.game.get_state()
        
        # Get initial info
        info = {
            'snake_info': self.game.get_snake_info(),
            'episode_steps': self.episode_steps,
            'episode_rewards': self.episode_rewards.copy(),
            'episode_scores': self.episode_scores.copy()
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, List[float], bool, bool, Dict]:
        """Execute one step in the environment"""
        # Validate action
        if len(action) != self.num_snakes:
            raise ValueError(f"Expected {self.num_snakes} actions, got {len(action)}")
        
        # Execute game step
        alive_status, rewards, game_over = self.game.step(action)
        
        # Update episode tracking
        self.episode_steps += 1
        for i in range(self.num_snakes):
            self.episode_rewards[i] += rewards[i]
            if self.game.snakes[i].alive:
                self.episode_scores[i] = self.game.snakes[i].score
        
        # Check if episode should terminate
        terminated = game_over
        truncated = self.episode_steps >= self.max_steps
        
        # Get new observation
        observation = self.game.get_state()
        
        # Prepare info
        info = {
            'snake_info': self.game.get_snake_info(),
            'episode_steps': self.episode_steps,
            'episode_rewards': self.episode_rewards.copy(),
            'episode_scores': self.episode_scores.copy(),
            'alive_status': alive_status,
            'game_over': game_over
        }
        
        # Scale rewards if specified
        if self.reward_scale != 1.0:
            rewards = [r * self.reward_scale for r in rewards]
        
        return observation, rewards, terminated, truncated, info
    
    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """Render the environment"""
        if mode is None:
            mode = self.render_mode
        
        if mode == "human":
            return self._render_human()
        elif mode == "rgb_array":
            return self._render_rgb_array()
        elif mode == "state":
            return self.game.get_state()
        else:
            raise ValueError(f"Unknown render mode: {mode}")
    
    def _render_human(self) -> None:
        """Render using pygame for human viewing"""
        if not hasattr(self, 'screen'):
            return
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Calculate cell size
        cell_size = min(800 // self.field_size, 600 // self.field_size)
        
        # Draw grid
        for i in range(self.field_size + 1):
            pygame.draw.line(self.screen, (50, 50, 50), 
                           (i * cell_size, 0), (i * cell_size, self.field_size * cell_size))
            pygame.draw.line(self.screen, (50, 50, 50), 
                           (0, i * cell_size), (self.field_size * cell_size, i * cell_size))
        
        # Draw snakes
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 128, 255), (255, 128, 128)]
        
        for snake in self.game.snakes:
            if snake.alive:
                color = colors[snake.id % len(colors)]
                
                # Draw body
                for segment in snake.body:
                    x, y = segment[1] * cell_size, segment[0] * cell_size
                    pygame.draw.rect(self.screen, color, 
                                   (x + 1, y + 1, cell_size - 2, cell_size - 2))
                
                # Draw head (slightly larger)
                head = snake.get_head()
                x, y = head[1] * cell_size, head[0] * cell_size
                pygame.draw.rect(self.screen, (255, 255, 255), 
                               (x + 2, y + 2, cell_size - 4, cell_size - 4))
        
        # Draw food
        for food_pos in self.game.food:
            x, y = food_pos[1] * cell_size, food_pos[0] * cell_size
            pygame.draw.circle(self.screen, (255, 255, 0), 
                             (x + cell_size // 2, y + cell_size // 2), cell_size // 3)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array for programmatic use"""
        # Get the state representation
        state = self.game.get_state()
        
        # Convert to RGB (3 channels)
        rgb = np.zeros((self.field_size, self.field_size, 3), dtype=np.uint8)
        
        # Snake bodies (red channel)
        rgb[:, :, 0] = (state[:, :, 0] * 255).astype(np.uint8)
        
        # Snake heads (green channel)
        rgb[:, :, 1] = (state[:, :, 1] * 255).astype(np.uint8)
        
        # Food (blue channel)
        rgb[:, :, 2] = (state[:, :, 2] * 255).astype(np.uint8)
        
        # Resize for better visualization
        rgb = cv2.resize(rgb, (self.field_size * 20, self.field_size * 20), 
                        interpolation=cv2.INTER_NEAREST)
        
        return rgb
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'screen'):
            pygame.quit()
    
    def get_legal_actions(self, snake_id: int) -> List[int]:
        """Get legal actions for a specific snake (avoiding 180-degree turns)"""
        if snake_id >= len(self.game.snakes):
            return [0, 1, 2, 3]
        
        snake = self.game.snakes[snake_id]
        if not snake.alive:
            return [0, 1, 2, 3]
        
        current_direction = snake.direction.value
        legal_actions = []
        
        for action in [0, 1, 2, 3]:
            # Check if action would cause 180-degree turn
            if abs(action - current_direction) != 2:
                legal_actions.append(action)
        
        return legal_actions if legal_actions else [0, 1, 2, 3]
    
    def get_snake_observation(self, snake_id: int) -> np.ndarray:
        """Get observation from the perspective of a specific snake"""
        if snake_id >= len(self.game.snakes):
            return self.game.get_state()
        
        # Get the global state
        global_state = self.game.get_state()
        
        # Create snake-specific observation
        # This could include additional information like:
        # - Distance to food
        # - Distance to other snakes
        # - Current direction
        # - Health/score information
        
        # For now, return the global state
        # In a more sophisticated implementation, you could add channels for:
        # - Relative positions
        # - Danger zones
        # - Strategic information
        
        return global_state
    
    def get_game_stats(self) -> Dict:
        """Get comprehensive game statistics"""
        return {
            'field_size': self.field_size,
            'num_snakes': self.num_snakes,
            'food_count': len(self.game.food),
            'episode_steps': self.episode_steps,
            'game_over': self.game.game_over,
            'snake_info': self.game.get_snake_info(),
            'episode_rewards': self.episode_rewards.copy(),
            'episode_scores': self.episode_scores.copy(),
            'alive_count': sum(1 for snake in self.game.snakes if snake.alive)
        }

# Register the environment
gym.register(
    id='snake_multiplayer-v0',
    entry_point='snake_env:SnakeMultiplayerEnv',
    max_episode_steps=1000,
)
