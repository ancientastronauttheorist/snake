import os
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pygame

# -----------------------------------------------------------------------------
# Device selection: Use CUDA if available, otherwise MPS if available, otherwise CPU.
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

# -----------------------------------------------------------------------------
# Helper: Linear interpolation between two colors.
# -----------------------------------------------------------------------------
def lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB colors."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

# -----------------------------------------------------------------------------
# Function to count trainable parameters.
# -----------------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------------------------------------------------------------
# Helper: Check if a position is valid (within bounds and not in snake body)
# -----------------------------------------------------------------------------
def is_valid_position(pos, snake_body, grid_size):
    x, y = pos
    n, m = grid_size
    if not (0 <= x < n and 0 <= y < m):
        return False
    if pos in snake_body:
        return False
    return True

# =============================================================================
# Snake Game Environment (Grid size 6x6) with updated reward shaping.
# =============================================================================
class SnakeGame:
    def __init__(self, n=6, m=6):
        self.n = n
        self.m = m
        self.grid = np.zeros((self.n, self.m), dtype=np.int32)
        self.reset()

    def reset(self):
        self.grid.fill(0)
        # Initialize reward-shaping tracking attributes.
        self.previous_path_length = float('inf')
        self.previous_manhattan = float('inf')
        self.no_path_counter = 0
        self.previous_free_cells = 0

        # Reset additional attributes.
        self.away_moves_counter = 0
        self.space_creation_steps = 0

        # Place snake in the middle with length 3 going upward.
        center = (self.n // 2, self.m // 2)
        head = (center[0] - 1, center[1])
        body = center
        tail = (center[0] + 1, center[1])
        self.snake = deque([head, body, tail])
        
        # Encode snake on the grid.
        snake_list = list(self.snake)
        for i, pos in enumerate(reversed(snake_list)):
            if i == len(snake_list) - 1:
                self.grid[pos] = 2  # Head
            else:
                self.grid[pos] = self.n * self.m + 1 - i

        self.place_food()
        self.steps_since_food = 0  # Steps taken without eating.
        self.done = False
        self.won = False  # Flag for win condition.
        self.score = len(self.snake)
        self._ate_food = False
        return self.get_state()

    def place_food(self):
        empty = list(zip(*np.where(self.grid == 0)))
        if empty:
            self.food = random.choice(empty)
            self.grid[self.food] = 1  # Food is encoded as 1.
        else:
            self.food = None
            if len(self.snake) == self.n * self.m:
                self.won = True
                self.done = True
                print("GAME WON! Snake filled the entire grid!")

    # ------------------------- Properties for reward function ------------------
    @property
    def head_position(self):
        return self.snake[0]

    @property
    def food_position(self):
        return self.food

    @property
    def snake_body(self):
        return list(self.snake)

    @property
    def grid_size(self):
        return (self.n, self.m)

    @property
    def ate_food(self):
        return self._ate_food

    @property
    def game_over(self):
        return self.done

    # -----------------------------------------------------------------------------
    # One-hot encoding of grid state.
    # -----------------------------------------------------------------------------
    def get_state(self):
        num_classes = self.n * self.m + 2
        state_onehot = np.eye(num_classes, dtype=np.float32)[self.grid]
        return state_onehot.flatten()

    # -----------------------------------------------------------------------------
    # Helper: Find the shortest path from start to goal using BFS.
    # -----------------------------------------------------------------------------
    def find_shortest_path(self, start, goal, snake_body):
        if goal is None:
            return False, float('inf')
        from collections import deque
        queue = deque()
        queue.append((start, 0))
        visited = set()
        obstacles = set(snake_body)
        while queue:
            pos, dist = queue.popleft()
            if pos == goal:
                return True, dist
            for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (pos[0] + move[0], pos[1] + move[1])
                if 0 <= new_pos[0] < self.n and 0 <= new_pos[1] < self.m:
                    if new_pos in visited:
                        continue
                    if new_pos in obstacles and new_pos != goal:
                        continue
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        return False, float('inf')

    # -----------------------------------------------------------------------------
    # Helper: Count reachable cells (using BFS) from a given start.
    # -----------------------------------------------------------------------------
    def count_reachable_cells(self, start, obstacles):
        from collections import deque
        queue = deque([start])
        visited = set([start])
        count = 0
        while queue:
            pos = queue.popleft()
            count += 1
            for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (pos[0] + move[0], pos[1] + move[1])
                if 0 <= new_pos[0] < self.n and 0 <= new_pos[1] < self.m:
                    if new_pos in visited or new_pos in obstacles:
                        continue
                    visited.add(new_pos)
                    queue.append(new_pos)
        return count

    # -----------------------------------------------------------------------------
    # Helper: Predict future body (assumes tail will move off after lookahead_steps).
    # -----------------------------------------------------------------------------
    def predict_future_body(self, snake_body, lookahead_steps):
        if lookahead_steps >= len(snake_body):
            return []
        return snake_body[:-lookahead_steps]

    # -----------------------------------------------------------------------------
    # Reward Function
    # -----------------------------------------------------------------------------
    def calculate_reward(self, action):
        head_pos = self.head_position
        food_pos = self.food_position
        snake_body = self.snake_body
        current_length = len(snake_body)
        grid_size = self.grid_size
        max_possible_length = grid_size[0] * grid_size[1]
        
        reward = 0.0
        if self.ate_food:
            food_reward_scale = min(1.0 + (current_length / max_possible_length), 2.0)
            reward += 1.0 * food_reward_scale

        if self.game_over and not self.won:
            progress = current_length / max_possible_length
            death_penalty = 1.0 + (progress * 0.5)
            reward -= death_penalty
            return reward
        
        if food_pos is not None:
            path_exists, path_length = self.find_shortest_path(head_pos, food_pos, snake_body)
            manhattan_dist = abs(head_pos[0] - food_pos[0]) + abs(head_pos[1] - food_pos[1])
        else:
            path_exists = False
            path_length = float('inf')
            manhattan_dist = float('inf')
        
        step_penalty = 0.01 * (1.0 + (current_length / max_possible_length * 0.5))
        reward -= step_penalty
        
        previous_path_length = getattr(self, 'previous_path_length', float('inf'))
        previous_manhattan = getattr(self, 'previous_manhattan', float('inf'))
        
        small_phase = current_length < 0.25 * max_possible_length
        medium_phase = 0.25 * max_possible_length <= current_length < 0.6 * max_possible_length
        
        if path_exists:
            self.no_path_counter = 0
            if previous_path_length != float('inf'):
                path_improvement = previous_path_length - path_length
                if small_phase:
                    reward += 0.2 * path_improvement
                elif medium_phase:
                    reward += 0.15 * path_improvement
                else:
                    reward += 0.1 * path_improvement
            self.previous_path_length = path_length
            self.previous_manhattan = manhattan_dist
        else:
            no_path_counter = getattr(self, 'no_path_counter', 0) + 1
            self.no_path_counter = no_path_counter
            
            free_cells = self.count_reachable_cells(head_pos, snake_body)
            previous_free_cells = getattr(self, 'previous_free_cells', 0)
            
            if no_path_counter > 10:
                lookahead_steps = min(5, len(snake_body) - 1)
                future_body = self.predict_future_body(snake_body, lookahead_steps)
                future_free_cells = self.count_reachable_cells(head_pos, future_body)
                
                cells_improvement = future_free_cells - previous_free_cells
                capped_improvement = max(min(cells_improvement, 5), -5)
                diminishing_factor = 1.0 / (1.0 + (no_path_counter / 20.0))
                reward += 0.1 * capped_improvement * diminishing_factor
                
                if food_pos is not None and manhattan_dist < previous_manhattan:
                    reward += 0.05
            
            self.previous_free_cells = free_cells
        
        if current_length > 5:
            head_x, head_y = head_pos
            safe_moves = 0
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (head_x + dx, head_y + dy)
                if is_valid_position(next_pos, snake_body[1:], grid_size):
                    safe_moves += 1
            safety_scale = 0.02 * (current_length / max_possible_length * 2)
            reward += safety_scale * safe_moves
        
        return reward

    # -----------------------------------------------------------------------------
    # Step Function
    # -----------------------------------------------------------------------------
    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {"score": self.score, "won": self.won}
        
        head = self.snake[0]
        if action == 0:
            move_vector = (-1, 0)
        elif action == 1:
            move_vector = (1, 0)
        elif action == 2:
            move_vector = (0, -1)
        elif action == 3:
            move_vector = (0, 1)
        else:
            move_vector = (0, 0)
        
        if len(self.snake) > 1:
            second = self.snake[1]
            current_direction = (head[0] - second[0], head[1] - second[1])
            if move_vector == (-current_direction[0], -current_direction[1]):
                self.done = True
                return self.get_state(), -1.0, True, {"score": self.score, "won": self.won}
        
        new_head = (head[0] + move_vector[0], head[1] + move_vector[1])
        if not (0 <= new_head[0] < self.n and 0 <= new_head[1] < self.m):
            self.done = True
            return self.get_state(), -1.0, True, {"score": self.score, "won": self.won}
        if new_head in self.snake and new_head != self.snake[-1]:
            self.done = True
            return self.get_state(), -1.0, True, {"score": self.score, "won": self.won}
        
        ate_food = (new_head == self.food)
        self.snake.appendleft(new_head)
        if ate_food:
            self.score = len(self.snake)
            self.steps_since_food = 0
            self._ate_food = True
        else:
            self.snake.pop()
            self.steps_since_food += 1
            self._ate_food = False
            self.score = len(self.snake)
        
        if not ate_food and self.steps_since_food > self.n * self.m:
            self.done = True
            return self.get_state(), -1.0, True, {"score": self.score, "won": self.won}
        
        if ate_food:
            self.place_food()
        
        reward = self.calculate_reward(action)
        self.grid.fill(0)
        if self.food is not None:
            self.grid[self.food] = 1
        snake_list = list(self.snake)
        for i, pos in enumerate(reversed(snake_list)):
            if i == len(snake_list) - 1:
                self.grid[pos] = 2  # Head
            else:
                self.grid[pos] = self.n * self.m + 1 - i
        
        return self.get_state(), reward, self.done, {"score": self.score, "won": self.won}

    # -----------------------------------------------------------------------------
    # Updated Render Function: Retro 80's Neon with Snake Gradient.
    # -----------------------------------------------------------------------------
    def render(self, screen, cell_size=80):
        # Fill background with pitch-black.
        screen.fill((0, 0, 0))
        # Draw grid lines.
        for row in range(self.n):
            for col in range(self.m):
                rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, (30, 30, 30), rect, 1)
        # Draw food (neon yellow).
        if self.food is not None:
            food_color = (255, 255, 0)
            row, col = self.food
            center = (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2)
            radius = cell_size // 3
            pygame.draw.circle(screen, food_color, center, radius)
        # Draw snake segments with a gradient from orange (head) to pink (tail).
        head_color = (255, 165, 0)   # Orange
        tail_color = (255, 105, 180) # Pink
        snake_list = list(self.snake)
        n_segments = len(snake_list)
        for i, pos in enumerate(snake_list):
            t = i / (n_segments - 1) if n_segments > 1 else 0
            color = lerp_color(head_color, tail_color, t)
            row, col = pos
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect, border_radius=8)

# =============================================================================
# Optimized PPO Neural Network
# =============================================================================
class PPO(nn.Module):
    def __init__(self, input_dim, output_dim=4):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.norm3 = nn.LayerNorm(64)
        self.policy_head = nn.Linear(64, output_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = F.relu(self.norm3(self.fc3(x)))
        return self.policy_head(x), self.value_head(x)

# =============================================================================
# Improved PPO Agent
# =============================================================================
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, clip_param=0.2, gamma=0.99, lam=0.95):
        self.model = PPO(state_dim, output_dim=action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

    def select_action(self, state, deterministic=False):
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).to(device)
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            probs = F.softmax(policy_logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs).item()
                return action, None, value
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action), value

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            last_gae = delta + self.gamma * self.lam * mask * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, old_log_probs, rewards, values, dones, next_value):
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions).to(device)
        old_log_probs = torch.stack(old_log_probs).detach().to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        values = torch.stack(values).detach().squeeze().to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = states.size(0)
        minibatch_size = batch_size // 4
        if minibatch_size < 1:
            minibatch_size = 1

        policy_losses, value_losses, entropy_losses = [], [], []
        for _ in range(4):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_indices = indices[start:end]
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                policy_logits, value_estimates = self.model(mb_states)
                probs = F.softmax(policy_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * mb_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value_estimates.squeeze(), mb_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                
        self.scheduler.step()
        self.entropy_coef = max(0.01, self.entropy_coef * 0.995)
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses)
        }

# =============================================================================
# Demo: Play one game using current model weights in a Pygame window.
# =============================================================================
def play_game_once(model_path="snake_model_optimized.pt"):
    pygame.init()
    cell_size = 80
    grid_size = (6, 6)
    text_area = 120  # Increased text area for the bigger window.
    screen_width = grid_size[1] * cell_size
    screen_height = grid_size[0] * cell_size + text_area
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("PPO Snake AI - Demo (Retro Neon)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28)

    env = SnakeGame(n=grid_size[0], m=grid_size[1])
    state_dim = env.n * env.m * (env.n * env.m + 2)
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim)
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded model from", model_path, "for demo game")
    else:
        print("No trained model found for demo!")
        pygame.quit()
        return

    action_mapping = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    state = env.reset()
    done = False
    current_action = ""
    current_reward = 0.0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        action, _, value = agent.select_action(state, deterministic=True)
        current_action = action_mapping.get(action, "")
        state, reward, done, info = env.step(action)
        current_reward = reward

        env.render(screen, cell_size=cell_size)
        text_rect = pygame.Rect(0, grid_size[0] * cell_size, screen_width, text_area)
        pygame.draw.rect(screen, (0, 0, 0), text_rect)
        if info.get("won", False):
            score_text = font.render(f"GAME WON! Score: {info.get('score', 0)}", True, (0, 255, 0))
        else:
            score_text = font.render(f"Score: {info.get('score', 0)}", True, (255, 255, 255))
        action_text = font.render(f"Action: {current_action}", True, (200, 200, 200))
        reward_text = font.render(f"Reward: {current_reward:.2f}", True, (200, 200, 200))
        
        screen.blit(score_text, (10, grid_size[0] * cell_size + 10))
        screen.blit(action_text, (10, grid_size[0] * cell_size + 45))
        screen.blit(reward_text, (10, grid_size[0] * cell_size + 80))
        pygame.display.flip()
        clock.tick(4)
    pygame.time.delay(1000)
    pygame.quit()

# =============================================================================
# Interactive Play Mode using the Trained PPO Model.
# =============================================================================
def play_game(model_path="snake_model_optimized.pt"):
    pygame.init()
    cell_size = 80
    grid_size = (6, 6)
    text_area = 120
    screen_width = grid_size[1] * cell_size
    screen_height = grid_size[0] * cell_size + text_area
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("PPO Snake AI (Retro Neon)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28)

    env = SnakeGame(n=grid_size[0], m=grid_size[1])
    state_dim = env.n * env.m * (env.n * env.m + 2)
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim)
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded model from", model_path)
    else:
        print("No trained model found!")
        return

    action_mapping = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    state = env.reset()
    running = True
    current_action = ""
    current_reward = 0.0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, _, _ = agent.select_action(state, deterministic=True)
        current_action = action_mapping.get(action, "")
        state, reward, done, info = env.step(action)
        current_reward = reward

        env.render(screen, cell_size=cell_size)
        text_rect = pygame.Rect(0, grid_size[0] * cell_size, screen_width, text_area)
        pygame.draw.rect(screen, (0, 0, 0), text_rect)
        if info.get("won", False):
            score_text = font.render(f"GAME WON! Score: {info.get('score', 0)}", True, (0, 255, 0))
        else:
            score_text = font.render(f"Score: {info.get('score', 0)}", True, (255, 255, 255))
        action_text = font.render(f"Action: {current_action}", True, (200, 200, 200))
        reward_text = font.render(f"Reward: {current_reward:.2f}", True, (200, 200, 200))
        
        screen.blit(score_text, (10, grid_size[0] * cell_size + 10))
        screen.blit(action_text, (10, grid_size[0] * cell_size + 45))
        screen.blit(reward_text, (10, grid_size[0] * cell_size + 80))
        pygame.display.flip()
        clock.tick(4)
        if done:
            pygame.time.delay(1000)
            state = env.reset()
    pygame.quit()

# =============================================================================
# Functions to save and load training metadata.
# =============================================================================
def save_training_metadata(metadata, filename="training_metadata.json"):
    import json
    with open(filename, 'w') as f:
        json.dump(metadata, f)
    print(f"Training metadata saved to {filename}")

def load_training_metadata(filename="training_metadata.json"):
    import json
    default_metadata = {
        "episodes_trained": 0,
        "best_avg_score": 0,
        "highest_score": 0,
        "wins": 0,
        "recent_scores": [],
        "recent_lengths": []
    }
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded training metadata from {filename}")
            return metadata
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return default_metadata
    else:
        print("No training metadata found, starting fresh")
        return default_metadata

# =============================================================================
# Improved Training Loop with Experience Buffer.
# =============================================================================
def train(total_episodes=1000000, update_freq=10, save_interval=100, play_demo=True, resume=True):
    env = SnakeGame(n=6, m=6)
    state_dim = env.n * env.m * (env.n * env.m + 2)
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim)
    model_file = "snake_model_optimized.pt"
    best_model_file = "snake_model_best.pt"
    win_model_file = "snake_model_win.pt"
    metadata_file = "training_metadata.json"
    
    if os.path.exists(model_file):
        agent.model.load_state_dict(torch.load(model_file, map_location=device))
        print("Loaded existing model from", model_file)
    
    print("Total model parameters:", count_parameters(agent.model))
    
    metadata = load_training_metadata(metadata_file) if resume else {
        "episodes_trained": 0,
        "best_avg_score": 0,
        "highest_score": 0,
        "wins": 0,
        "recent_scores": [],
        "recent_lengths": []
    }
    
    scores = metadata["recent_scores"][-1000:] if "recent_scores" in metadata else []
    episode_lengths = metadata["recent_lengths"][-1000:] if "recent_lengths" in metadata else []
    wins = metadata["wins"] if "wins" in metadata else 0
    highest_score = metadata["highest_score"] if "highest_score" in metadata else 0
    best_avg_score = metadata["best_avg_score"] if "best_avg_score" in metadata else 0
    start_episode = metadata["episodes_trained"] if "episodes_trained" in metadata else 0
    
    print(f"Resuming from episode {start_episode} | Best avg score: {best_avg_score:.2f} | Wins: {wins}")
    
    buffer_states = []
    buffer_actions = []
    buffer_log_probs = []
    buffer_rewards = []
    buffer_values = []
    buffer_dones = []
    
    episode = start_episode
    while episode < start_episode + total_episodes:
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            buffer_states.append(torch.tensor(state.flatten(), dtype=torch.float32))
            buffer_actions.append(action)
            buffer_log_probs.append(log_prob if log_prob is not None else torch.tensor(0.0).to(device))
            buffer_values.append(value.squeeze())
            next_state, reward, done, info = env.step(action)
            buffer_rewards.append(reward)
            buffer_dones.append(done)
            state = next_state
            episode_reward += reward
            step_count += 1
        
        episode += 1
        scores.append(info.get("score", 0))
        episode_lengths.append(step_count)
        
        if info.get("won", False):
            wins += 1
            print(f"WIN #{wins} at episode {episode}! Score: {info.get('score', 0)}")
            torch.save(agent.model.state_dict(), win_model_file)
            print(f"Saved winning model to {win_model_file}")
        
        if len(buffer_states) >= update_freq * 200 or episode % update_freq == 0:
            last_value = torch.tensor(0.0).to(device) if done else None
            if not done:
                with torch.no_grad():
                    _, last_value = agent.model(torch.tensor(state.flatten(), dtype=torch.float32).to(device))
            losses = agent.update(
                buffer_states, 
                buffer_actions, 
                buffer_log_probs, 
                buffer_rewards, 
                buffer_values, 
                buffer_dones,
                last_value
            )
            buffer_states, buffer_actions, buffer_log_probs = [], [], []
            buffer_rewards, buffer_values, buffer_dones = [], [], []
            if episode % 10 == 0:
                last_100_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                last_100_len = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
                print(f"Episode: {episode} | Avg Score: {last_100_avg:.2f} | Avg Length: {last_100_len:.2f} | Best: {highest_score} | Wins: {wins}")
                print(f"Losses: Policy={losses['policy_loss']:.4f}, Value={losses['value_loss']:.4f}, Entropy={losses['entropy']:.4f}")
        
        if episode % save_interval == 0:
            last_100_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            torch.save(agent.model.state_dict(), model_file)
            print(f"Saved model to {model_file}")
            if last_100_avg > best_avg_score:
                best_avg_score = last_100_avg
                torch.save(agent.model.state_dict(), best_model_file)
                print(f"New best model! Avg score: {best_avg_score:.2f}")
            metadata = {
                "episodes_trained": episode,
                "best_avg_score": best_avg_score,
                "highest_score": highest_score,
                "wins": wins,
                "recent_scores": scores[-1000:],
                "recent_lengths": episode_lengths[-1000:]
            }
            save_training_metadata(metadata)
        
        if info.get("score", 0) > highest_score:
            highest_score = info.get("score", 0)
        
        if play_demo and episode % 1000 == 0:
            print("Demo: Playing one game with current model weights...")
            play_game_once(model_file)
    return agent

# =============================================================================
# Main: Choose between training and playing.
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "play"],
                        help="Choose to train the model or watch it play.")
    parser.add_argument("--episodes", type=int, default=1000000,
                        help="Number of training episodes (only used in train mode).")
    parser.add_argument("--no_demo", action="store_true",
                        help="If set, do not play a demo game every 1,000 episodes during training.")
    parser.add_argument("--update_freq", type=int, default=10,
                        help="How often to update the policy (in episodes).")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="How often to save the model (in episodes).")
    parser.add_argument("--model_path", type=str, default="snake_model_best.pt",
                        help="Path to the model file to use when playing.")
    parser.add_argument("--no_resume", action="store_true",
                        help="If set, start training from scratch instead of resuming.")
    args = parser.parse_args()

    play_demo = not args.no_demo
    resume = not args.no_resume
    state_dim = 6 * 6 * (6 * 6 + 2)
    action_dim = 4
    temp_agent = PPOAgent(state_dim, action_dim)
    print("Total model parameters:", count_parameters(temp_agent.model))

    if args.mode == "train":
        train(
            total_episodes=args.episodes,
            play_demo=play_demo,
            update_freq=args.update_freq,
            save_interval=args.save_interval,
            resume=resume
        )
    else:
        play_game(args.model_path)
