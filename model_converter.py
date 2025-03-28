#!/usr/bin/env python3
"""
Snake Game Model Converter

This script provides utilities to:
1. Convert a trained PyTorch model to a portable NumPy format (.npz)
2. Load a portable model and optimize it for the current hardware (CUDA, MPS/M1, or CPU)

Usage:
    # Export PyTorch model to portable format
    python model_converter.py export --input snake_model_best.pt --output snake_model_portable.npz
    
    # Import portable model and optimize for current hardware
    python model_converter.py import --input snake_model_portable.npz --output snake_model_optimized.pt
    
    # Run the game with the optimized model
    python model_converter.py play --input snake_model_optimized.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------------------------------------------------------
# Define the PPO model architecture (must match the original model)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Helper: Detect the best available device
# -----------------------------------------------------------------------------
def get_optimal_device():
    """Detect and return the best available device for tensor operations."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# -----------------------------------------------------------------------------
# Export a PyTorch model to portable format
# -----------------------------------------------------------------------------
def export_model(input_path, output_path):
    """
    Export a PyTorch model to a portable NumPy format.
    
    Args:
        input_path: Path to the PyTorch model (.pt file)
        output_path: Path to save the portable model (.npz file)
    """
    # Grid size from the original code
    grid_size = (6, 6)
    state_dim = grid_size[0] * grid_size[1] * (grid_size[0] * grid_size[1] + 2)
    action_dim = 4
    
    # Create a new model with the same architecture
    model = PPO(state_dim, action_dim)
    
    # Load the state dict from the input model (to CPU to avoid device issues)
    try:
        state_dict = torch.load(input_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {input_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Extract weights and biases as NumPy arrays
    weights_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy()
    
    # Add model metadata
    weights_dict['_metadata'] = np.array([
        state_dim,           # Input dimension
        action_dim,          # Output dimension
        grid_size[0],        # Grid height
        grid_size[1]         # Grid width
    ])
    
    # Save as NumPy arrays
    try:
        np.savez(output_path, **weights_dict)
        print(f"Model successfully exported to {output_path}")
    except Exception as e:
        print(f"Error saving portable model: {e}")
        sys.exit(1)

# -----------------------------------------------------------------------------
# Import a portable model and optimize for the current hardware
# -----------------------------------------------------------------------------
def import_model(input_path, output_path):
    """
    Import a portable model and optimize it for the current hardware.
    
    Args:
        input_path: Path to the portable model (.npz file)
        output_path: Path to save the optimized PyTorch model (.pt file)
    """
    device = get_optimal_device()
    
    # Load the portable model
    try:
        weights = np.load(input_path)
        print(f"Successfully loaded portable model from {input_path}")
    except Exception as e:
        print(f"Error loading portable model: {e}")
        sys.exit(1)
    
    # Extract metadata if available, otherwise use defaults
    if '_metadata' in weights:
        metadata = weights['_metadata']
        state_dim = int(metadata[0])
        action_dim = int(metadata[1])
        print(f"Using model metadata: state_dim={state_dim}, action_dim={action_dim}")
    else:
        # Default values from the original code
        grid_size = (6, 6)
        state_dim = grid_size[0] * grid_size[1] * (grid_size[0] * grid_size[1] + 2)
        action_dim = 4
        print(f"No metadata found, using defaults: state_dim={state_dim}, action_dim={action_dim}")
    
    # Create a new model with the same architecture
    model = PPO(state_dim, action_dim).to(device)
    
    # Apply optimization techniques based on the device
    if device.type == 'cuda':
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.cuda, 'amp'):
            print("Enabling automatic mixed precision for CUDA")
            model = torch.cuda.amp.autocast(enabled=True)(model)
    
    # Create a state dict to load the weights (skip metadata)
    state_dict = {}
    for name in weights.files:
        if name != '_metadata':
            param = torch.tensor(weights[name], device=device)
            state_dict[name] = param
    
    # Load the weights into the model
    try:
        model.load_state_dict(state_dict)
        print("Weights successfully loaded into model")
    except Exception as e:
        print(f"Error loading weights into model: {e}")
        sys.exit(1)
    
    # Save the optimized model
    try:
        torch.save(model.state_dict(), output_path)
        print(f"Optimized model successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving optimized model: {e}")
        sys.exit(1)

# -----------------------------------------------------------------------------
# Simple function to test the model by loading and running inference
# -----------------------------------------------------------------------------
def test_model(model_path):
    """Load a model and perform a simple inference test."""
    device = get_optimal_device()
    
    # Determine if it's a portable or PyTorch model
    is_portable = model_path.endswith('.npz')
    
    if is_portable:
        # Load portable model
        try:
            weights = np.load(model_path)
            if '_metadata' in weights:
                metadata = weights['_metadata']
                state_dim = int(metadata[0])
                action_dim = int(metadata[1])
            else:
                grid_size = (6, 6)
                state_dim = grid_size[0] * grid_size[1] * (grid_size[0] * grid_size[1] + 2)
                action_dim = 4
            
            # Create model
            model = PPO(state_dim, action_dim).to(device)
            
            # Load weights
            state_dict = {}
            for name in weights.files:
                if name != '_metadata':
                    param = torch.tensor(weights[name], device=device)
                    state_dict[name] = param
            
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading portable model: {e}")
            sys.exit(1)
    else:
        # Load PyTorch model
        try:
            grid_size = (6, 6)
            state_dim = grid_size[0] * grid_size[1] * (grid_size[0] * grid_size[1] + 2)
            action_dim = 4
            
            model = PPO(state_dim, action_dim).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            sys.exit(1)
    
    # Create a dummy input and run inference
    dummy_input = torch.rand(1, state_dim, device=device)
    
    # Measure inference time
    import time
    start_time = time.time()
    num_iterations = 100
    
    with torch.no_grad():
        for _ in range(num_iterations):
            policy, value = model(dummy_input)
    
    avg_time = (time.time() - start_time) / num_iterations
    print(f"Model loaded successfully!")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Policy output shape: {policy.shape}, Value output shape: {value.shape}")

# -----------------------------------------------------------------------------
# Play game using the model (minimal implementation)
# -----------------------------------------------------------------------------
def play_game(model_path):
    """
    Play the snake game using the specified model.
    This is a minimal implementation that will attempt to import the necessary components
    from the original snake game code.
    """
    try:
        # Try to import from the local snake game module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # First, try to import directly if running from the same directory
        try:
            from __main__ import SnakeGame, play_game as original_play_game
            print("Successfully imported game components from main module")
            
            # If the original play_game function exists, use it
            if model_path.endswith('.npz'):
                print("Converting portable model to optimized format first...")
                temp_path = "temp_optimized_model.pt"
                import_model(model_path, temp_path)
                original_play_game(temp_path)
            else:
                original_play_game(model_path)
            return
        except ImportError:
            pass
        
        # Otherwise, look for possible module names based on common filenames
        for module_name in ["snake_game", "snake"]:
            try:
                module = __import__(module_name)
                if hasattr(module, "play_game") and hasattr(module, "SnakeGame"):
                    print(f"Found game components in {module_name}")
                    
                    if model_path.endswith('.npz'):
                        print("Converting portable model to optimized format first...")
                        temp_path = "temp_optimized_model.pt"
                        import_model(model_path, temp_path)
                        module.play_game(temp_path)
                    else:
                        module.play_game(model_path)
                    return
            except ImportError:
                continue
        
        # If we get here, we couldn't find the game components
        print("Could not locate the original snake game module.")
        print("Please run the game from the original snake game file:")
        print("1. Convert your model: python model_converter.py import --input portable_model.npz --output optimized_model.pt")
        print("2. Run the game: python snake_game.py --mode play --model_path optimized_model.pt")
    
    except Exception as e:
        print(f"Error during play: {e}")
        sys.exit(1)

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Snake Game Model Converter")
    
    # Main command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export PyTorch model to portable format")
    export_parser.add_argument("--input", type=str, required=True, help="Input PyTorch model path (.pt)")
    export_parser.add_argument("--output", type=str, required=True, help="Output portable model path (.npz)")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import portable model and optimize for current hardware")
    import_parser.add_argument("--input", type=str, required=True, help="Input portable model path (.npz)")
    import_parser.add_argument("--output", type=str, required=True, help="Output optimized model path (.pt)")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test a model by loading and running inference")
    test_parser.add_argument("--input", type=str, required=True, help="Model path (.pt or .npz)")
    
    # Play command
    play_parser = subparsers.add_parser("play", help="Play the game using a model")
    play_parser.add_argument("--input", type=str, required=True, help="Model path (.pt or .npz)")
    
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "export":
        export_model(args.input, args.output)
    elif args.command == "import":
        import_model(args.input, args.output)
    elif args.command == "test":
        test_model(args.input)
    elif args.command == "play":
        play_game(args.input)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
