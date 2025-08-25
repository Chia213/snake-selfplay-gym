import numpy as np
import random
from typing import List, Tuple, Optional, Dict
from enum import Enum

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Snake:
    def __init__(self, snake_id: int, start_pos: Tuple[int, int], start_direction: Direction = Direction.RIGHT):
        self.id = snake_id
        self.body = [start_pos]
        self.direction = start_direction
        self.alive = True
        self.score = 0
        self.growth_pending = 0
        
    def move(self, new_direction: Optional[Direction] = None) -> Tuple[int, int]:
        """Move the snake and return the new head position"""
        if new_direction is not None:
            # Prevent 180-degree turns
            if abs(new_direction.value - self.direction.value) != 2:
                self.direction = new_direction
        
        head = self.body[0]
        if self.direction == Direction.UP:
            new_head = (head[0] - 1, head[1])
        elif self.direction == Direction.RIGHT:
            new_head = (head[0], head[1] + 1)
        elif self.direction == Direction.DOWN:
            new_head = (head[0] + 1, head[1])
        else:  # Direction.LEFT
            new_head = (head[0], head[1] - 1)
            
        # Add new head
        self.body.insert(0, new_head)
        
        # Remove tail unless growing
        if self.growth_pending > 0:
            self.growth_pending -= 1
        else:
            self.body.pop()
            
        return new_head
    
    def grow(self):
        """Make the snake grow by one segment"""
        self.growth_pending += 1
        self.score += 1
    
    def die(self):
        """Mark the snake as dead"""
        self.alive = False
    
    def get_head(self) -> Tuple[int, int]:
        """Get the current head position"""
        return self.body[0]
    
    def get_body(self) -> List[Tuple[int, int]]:
        """Get all body positions"""
        return self.body.copy()

class SnakeGame:
    def __init__(self, field_size: int = 20, num_snakes: int = 2, food_count: int = 3):
        self.field_size = field_size
        self.num_snakes = num_snakes
        self.food_count = food_count
        self.snakes: List[Snake] = []
        self.food: List[Tuple[int, int]] = []
        self.game_over = False
        self.step_count = 0
        self.max_steps = field_size * field_size * 2  # Prevent infinite games
        
        self._initialize_game()
    
    def _initialize_game(self):
        """Initialize the game with snakes and food"""
        # Initialize snakes at different starting positions
        start_positions = [
            (self.field_size // 4, self.field_size // 4),
            (3 * self.field_size // 4, 3 * self.field_size // 4),
            (self.field_size // 4, 3 * self.field_size // 4),
            (3 * self.field_size // 4, self.field_size // 4)
        ]
        
        start_directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        
        for i in range(self.num_snakes):
            pos = start_positions[i % len(start_positions)]
            direction = start_directions[i % len(start_directions)]
            snake = Snake(i, pos, direction)
            self.snakes.append(snake)
        
        # Spawn initial food
        self._spawn_food()
    
    def _spawn_food(self):
        """Spawn food at random empty positions"""
        while len(self.food) < self.food_count:
            # Find empty positions
            empty_positions = []
            for i in range(self.field_size):
                for j in range(self.field_size):
                    pos = (i, j)
                    if not self._is_position_occupied(pos):
                        empty_positions.append(pos)
            
            if empty_positions:
                food_pos = random.choice(empty_positions)
                self.food.append(food_pos)
            else:
                break  # No empty positions left
    
    def _is_position_occupied(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is occupied by a snake or food"""
        # Check snakes
        for snake in self.snakes:
            if snake.alive and pos in snake.body:
                return True
        
        # Check food
        if pos in self.food:
            return True
        
        return False
    
    def _check_collision(self, snake: Snake, new_head: Tuple[int, int]) -> bool:
        """Check if a snake collides with walls, other snakes, or itself"""
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.field_size or 
            new_head[1] < 0 or new_head[1] >= self.field_size):
            return True
        
        # Self collision (check if new head collides with body)
        if new_head in snake.body[1:]:
            return True
        
        # Other snake collision
        for other_snake in self.snakes:
            if other_snake.id != snake.id and other_snake.alive:
                if new_head in other_snake.body:
                    return True
        
        return False
    
    def _check_food_collision(self, snake: Snake) -> bool:
        """Check if a snake ate food and handle it"""
        head = snake.get_head()
        if head in self.food:
            self.food.remove(head)
            snake.grow()
            self._spawn_food()
            return True
        return False
    
    def step(self, actions: List[int]) -> Tuple[List[bool], List[float], bool]:
        """Execute one game step with the given actions"""
        if self.game_over:
            return [False] * self.num_snakes, [0.0] * self.num_snakes, True
        
        self.step_count += 1
        
        # Convert action indices to directions
        directions = [Direction(action) for action in actions]
        
        # Move all snakes
        rewards = [0.0] * self.num_snakes
        alive_status = [snake.alive for snake in self.snakes]
        
        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                continue
                
            # Move snake
            new_head = snake.move(directions[i])
            
            # Check collisions
            if self._check_collision(snake, new_head):
                snake.die()
                alive_status[i] = False
                rewards[i] = -1.0  # Death penalty
            else:
                # Check food collision
                if self._check_food_collision(snake):
                    rewards[i] = 1.0  # Food reward
                else:
                    rewards[i] = -0.01  # Small step penalty to encourage efficiency
        
        # Check if game is over
        alive_count = sum(alive_status)
        if alive_count <= 1 or self.step_count >= self.max_steps:
            self.game_over = True
            
            # Give bonus rewards to surviving snakes
            for i, alive in enumerate(alive_status):
                if alive:
                    rewards[i] += 5.0  # Survival bonus
        
        # Convert to numpy arrays for compatibility
        alive_status = np.array(alive_status, dtype=np.bool_)
        rewards = np.array(rewards, dtype=np.float32)
        
        return alive_status, rewards, self.game_over
    
    def get_state(self) -> np.ndarray:
        """Get the current game state as a numpy array"""
        # Create a 3D array: [field_size, field_size, channels]
        # Channels: snake bodies, snake heads, food, walls
        state = np.zeros((self.field_size, self.field_size, 4), dtype=np.float32)
        
        # Add snake bodies (channel 0)
        for snake in self.snakes:
            if snake.alive:
                for segment in snake.body:
                    if 0 <= segment[0] < self.field_size and 0 <= segment[1] < self.field_size:
                        state[segment[0], segment[1], 0] = 1.0
        
        # Add snake heads (channel 1)
        for snake in self.snakes:
            if snake.alive:
                head = snake.get_head()
                if 0 <= head[0] < self.field_size and 0 <= head[1] < self.field_size:
                    state[head[0], head[1], 1] = 1.0
        
        # Add food (channel 2)
        for food_pos in self.food:
            if 0 <= food_pos[0] < self.field_size and 0 <= food_pos[1] < self.field_size:
                state[food_pos[0], food_pos[1], 2] = 1.0
        
        # Add walls (channel 3)
        state[0, :, 3] = 1.0  # Top wall
        state[-1, :, 3] = 1.0  # Bottom wall
        state[:, 0, 3] = 1.0  # Left wall
        state[:, -1, 3] = 1.0  # Right wall
        
        return state
    
    def get_snake_info(self) -> List[Dict]:
        """Get information about all snakes"""
        info = []
        for snake in self.snakes:
            info.append({
                'id': snake.id,
                'alive': snake.alive,
                'score': snake.score,
                'body_length': len(snake.body),
                'head': snake.get_head(),
                'direction': snake.direction.value
            })
        return info
    
    def reset(self):
        """Reset the game to initial state"""
        self.snakes = []
        self.food = []
        self.game_over = False
        self.step_count = 0
        self._initialize_game()
    
    def render(self) -> np.ndarray:
        """Render the game state as a visual array (for debugging)"""
        # Create a simple visual representation
        render = np.zeros((self.field_size, self.field_size), dtype=np.uint8)
        
        # Add snakes (value = snake_id + 1)
        for snake in self.snakes:
            if snake.alive:
                for segment in snake.body:
                    if 0 <= segment[0] < self.field_size and 0 <= segment[1] < self.field_size:
                        render[segment[0], segment[1]] = snake.id + 1
        
        # Add food (value = 255)
        for food_pos in self.food:
            if 0 <= food_pos[0] < self.field_size and 0 <= food_pos[1] < self.field_size:
                render[food_pos[0], food_pos[1]] = 255
        
        return render



