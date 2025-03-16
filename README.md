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

## Project Structure

```
snake/
├── snake.py             # Main code file with environment, agent and training logic
├── snake_model_best.pt  # Best performing model (created during training)
├── snake_model_optimized.pt # Latest model (created during training)
├── README.md            # This file
└── LICENSE              # License file
```

## Technical Details

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
