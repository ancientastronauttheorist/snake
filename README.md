# Snake AI with Proximal Policy Optimization (PPO)

A reinforcement learning implementation of the classic Snake game using Proximal Policy Optimization (PPO). This project demonstrates how to train an AI agent to play Snake at a high level through sophisticated reward engineering and modern deep reinforcement learning techniques.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Demo

TODO: Include a GIF or screenshot of the agent playing Snake

## Features

- **Advanced PPO Implementation**:
  - Neural network with policy and value heads
  - Generalized Advantage Estimation (GAE)
  - Policy clipping, entropy regularization, and learning rate scheduling
  - Efficient experience buffer for stable learning

- **Sophisticated Reward Function**:
  - Phase-based rewards that adapt to snake size
  - Path-following and space-creation strategies
  - Safety bonuses for ensuring future movement options
  - Intelligent lookahead to prevent self-trapping

- **6x6 Grid Snake Environment**:
  - Complete game mechanics with collision detection
  - One-hot state encoding for neural network processing
  - BFS path finding for shortest path calculations
  - Reachable cell counting for space evaluation

- **Visualization**:
  - Real-time game rendering with Pygame
  - Display of scores, actions, and rewards
  - Demo mode for watching AI play
  - Interactive mode for continuous gameplay

- **Model Portability & Optimization**:
  - Convert models between PyTorch (.pt) and portable NumPy (.npz) formats
  - Hardware-specific optimizations for CUDA, Apple MPS, and CPU
  - Share trained models across different systems and hardware configurations
  - Test inference performance on your hardware

## Installation

```bash
# Clone the repository
git clone https://github.com/ancientastronauttheorist/snake.git
cd snake

# Install dependencies
pip install torch numpy pygame
```

## Usage

The program supports two main modes:

```bash
# Train the agent from scratch
python snake_ppo.py --mode train --episodes 1000000

# Train with custom parameters
python snake_ppo.py --mode train --episodes 500000 --update_freq 20 --save_interval 200 --no_demo

# Watch the trained agent play
python snake_ppo.py --mode play
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Either "train" or "play" | "train" |
| `--episodes` | Number of training episodes | 1000000 |
| `--update_freq` | How often to update the policy (in episodes) | 10 |
| `--save_interval` | How often to save the model (in episodes) | 100 |
| `--no_demo` | If set, do not play demo games during training | False |

### Model Converter Tool

The project includes a model converter utility that helps with portability and optimization:

```bash
# Export PyTorch model to portable format
python model_converter.py export --input snake_model_best.pt --output snake_model_portable.npz

# Import portable model and optimize for current hardware
python model_converter.py import --input snake_model_portable.npz --output snake_model_optimized.pt

# Test model inference performance
python model_converter.py test --input snake_model_best.pt

# Run the game with a specific model
python model_converter.py play --input snake_model_optimized.pt
```

This tool is useful when:
- Sharing models across different hardware platforms (CUDA, Apple M-series, CPU)
- Deploying to systems without PyTorch installed
- Optimizing model performance for your specific hardware
- Testing inference speeds before deployment

## Project Structure

```
snake/
├── snake_ppo.py              # Main code file with environment, agent and training logic
├── model_converter.py        # Utility for model portability and optimization
├── snake_model_best.pt       # Best performing model (created during training)
├── snake_model_optimized.pt  # Latest model (created during training)
├── README.md                 # This file
└── LICENSE                   # License file
```

## Technical Details

### Reward Function Deep Dive

The heart of this implementation is the meticulously crafted reward function that enables sophisticated emergent behaviors. Unlike many simplistic RL Snake implementations that only reward food collection, this system uses a hierarchical, adaptive approach:

#### Phase-Based Adaptive Rewards
- **Small Snake Phase** (< 25% of grid): Emphasizes path optimization with higher path improvement rewards
- **Medium Snake Phase** (25-60% of grid): Balances between path optimization and safety
- **Large Snake Phase** (> 60% of grid): Prioritizes space management and safety over direct path following

#### Dual Strategy System
1. **Path-Following Strategy**
   - When a path to food exists, rewards are proportional to path length improvements
   - Rewards are scaled based on snake size to ensure appropriate risk-taking
   - Maintains "shortest path memory" to prevent oscillating behaviors

2. **Space-Creation Strategy**
   - Automatically engages when no path to food exists
   - Uses BFS to count reachable cells and rewards increasing accessible space
   - Incorporates "lookahead" that simulates future tail movement to avoid dead ends
   - Features diminishing returns based on time spent without a path to prevent endless loops

#### Safety and Efficiency Components
- **Safety Bonus**: Rewards having multiple safe movement options proportional to snake size
- **Time Pressure**: Small penalties for each step to encourage efficiency
- **Death Penalty**: Scales with progress to make later-game deaths more impactful
- **Food Reward**: Scales with progress to make later-game food more valuable

This reward function allows the agent to learn complex strategies including:
- Creating escape routes when trapped
- Efficient food collection patterns
- Wall-following behaviors when appropriate
- Curling techniques to maximize space utilization
- Self-avoidance without explicit programming

The reward values are carefully balanced to ensure that no single component dominates the learning process, resulting in an agent that can effectively navigate the full gameplay loop from small to large snake sizes.

### State Representation

The game state is encoded using a sophisticated grid representation that preserves both spatial information and temporal body ordering:

#### Grid Encoding

The 6×6 grid is represented internally as a 2D array with a consistent encoding scheme that creates a "temporal gradient" along the snake's body:

- **0**: Empty cell
- **1**: Food cell
- **2**: Snake head (always constant)
- **3 to n*m+1**: Snake body segments with a clever encoding pattern

The snake's body encoding uses a formula of `n*m + 1 - i` where:
- `n*m` is the total grid size (36 for a 6×6 grid)
- `i` is the position index when iterating from tail to head

This creates a consistent pattern where:
- The tail always starts at the highest value (n*m + 1 = 37)
- Each segment closer to the head decrements by 1
- The head is always 2

When the snake grows after eating food:
1. A new head is added (always value 2)
2. All existing body segments shift down and are re-encoded
3. The tail maintains the highest value

This clever encoding scheme provides several advantages:
- Direction information is implicitly encoded (gradient always flows tail→head)
- Agent can determine exact body configuration from a single frame
- Head and food detection remain constant regardless of snake length
- Encoding is invariant to snake growth, ensuring consistent learning

The "temporal gradient" enables the network to understand the snake's movement direction and potential future positions without requiring multiple time steps as input.

#### One-Hot Transformation

To create a format suitable for the neural network, the grid undergoes a one-hot encoding transformation:
1. Each cell value is converted to a binary vector of length (n*m+2) [38 for a 6×6 grid]
2. This creates a 3D tensor of shape (6, 6, 38)
3. The tensor is flattened to a 1D vector of length 1368

This encoding method provides several advantages:
- Preserves spatial relationships between grid elements
- Distinguishes between all possible snake body configurations
- Ensures equal neural network attention to all state elements
- Eliminates any implicit ordinal relationships between state values

The detailed state representation enables the agent to learn subtle patterns that would be impossible with simpler encodings, such as recognizing when the snake is about to trap itself or determining the optimal path to the food while avoiding its own body.

### Environment

- 6x6 grid for the Snake game
- Food placement with collision detection
- Step-based timeout to prevent infinite games
- Comprehensive state encoding for the neural network

### Neural Network Architecture

```
PPO(
  (fc1): Linear(in_features=1368, out_features=256)
  (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
  (fc2): Linear(in_features=256, out_features=128)
  (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (fc3): Linear(in_features=128, out_features=64)
  (norm3): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
  (policy_head): Linear(in_features=64, out_features=4)
  (value_head): Linear(in_features=64, out_features=1)
)
```

### Training Process

The agent is trained using PPO with the following key components:
- Experience buffer for stable batch updates
- Generalized Advantage Estimation (GAE)
- Separate policy and value function losses
- Entropy regularization to encourage exploration
- Learning rate scheduling
- Gradient clipping for stability

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI's PPO Paper](https://arxiv.org/abs/1707.06347)
- [PyTorch](https://pytorch.org/)
- [Pygame](https://www.pygame.org/)
