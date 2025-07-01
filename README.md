# Multi-Robot Exploration with RLlib

This is a Deep Reinforcement Learning (DRL) framework for coordinating a team of robots to explore an unknown grid-world environment while maintaining a connected communication network. The proposed system adapts a multi-agent actor-critic algorithm (Multi-Agent PPO) using Ray RLlib’s legacy API and PettingZoo (Gymnasium) for environment modeling. Inspired by a MATLAB-based area coverage example, this project extends the task to include communication constraints by modifying the reward function. Empirical results demonstrate that the learned policy achieves moderate to high success rates in preserving network connectivity during exploration, and shows potential for generalization to larger teams.

## Installation (Python v3.10)

**Install Pytorch with CUDA support (Required)**
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_ver}
```

**Install Dependencies (Python version 3.10 required for pre-trained models -- I use 3.10.11)**
```
pip install ray[tune]==2.44.1 PettingZoo==1.22.3 dm_tree scipy pygame matplotlib lz4 pyyaml
```

## Training Models

Additional arguments:
- --seed \<integer\> (defaults to 42)
- --centralized_critic (boolean flag to use global observation for value function -- experimental)

**General Command:**
```
python main.py --config <file_name> --model_name <model_name> --seed <seed>
```

**Example:**
```
python main.py --config default --model_name model_v1 --seed 42
```

## Testing Models

The following arguments are optional arguments: 
- --seed \<integer\> (defaults to 42)
- --num_test_episodes \<integer\> (defaults to 1) 

**General Command:**
```
python main.py --test --config <file_name> --model_name <model_name>
```

**Full Connectivity Example:**
```
python main.py --test --config s12_br10 --model_name s12_br10_d --seed 25 
```

**Testing Model Checkpoints**
```
python main.py --test --config rw5 --model_name ../ckpt/model_name/3 
```

## Project Structure and Configuration

```
/
│
├── config/
│   └── <config_files>             # YAML config files (e.g., default.yaml, custom.yaml)
│
├── env/
│   ├── obstacle_mats/
│   │   └── mat1                   # Custom obstacle maps (origin at top-left)
│   └── grid_world.py             # GridWorld environment definition
│
├── models/
│   ├── ckpt/
│   │   └── <model_name>/
│   │       └── <checkpoint_num>/ # Checkpointed model state (RLlib format)
│   │           └── <files>       # RLlib checkpoint files
│   ├── saved/
│   │   └── <model_name>/         # Saved models for evaluation
│   ├── cnn.py                    # CNN model architecture
│   └── rl_wrappers.py            # RLlib wrappers and custom model handling
│
├── main.py                       # Main entry point for training and testing
├── utils.py                      # Utilities: metrics saving, model building, CLI args
```
- Add new obstacle layouts under env/obstacle_mats/.
- Add custom configuration files under config/ and use --config <file_name> when running.
- Models are saved or checkpointed under models/saved/ and models/ckpt/, respectively.
- Metrics will be stored at the root '/' after training completes.

## Example configuration file format: 
```yaml
environment:
    map_name: mat1
    max_steps: 1000
    size: 12
    num_agents: 3
    cr: 5

reward_scheme:
    new_tile_connected: 2.0
    new_tile_disconnected: -10.0
    old_tile_connected: -0.1
    old_tile_disconnected: -10.0
    obstacle: -1.0
    terminated: 50

training:
    gamma: 0.9
    lr: 0.0001
    grad_clip: 1.0
    train_batch_size: 2000
    num_epochs: 5
    minibatch_size: 200
    l2_regularization: 0.0001
    num_iterations: 8000
    target_reward: 380
```
