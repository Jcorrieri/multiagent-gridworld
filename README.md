# Multi-Robot Exploration with RLlib

## Training Models

**General Command:**
```
python main.py --config <file_name> --model_name <model_name> --seed <seed>
```

**Example:**
```
python main.py --config default --model_name mppo_connect_v3 --seed 89
```

## Testing Models

**General Command:**
```
python main.py --test --config <file_name> --model_name <model_name> --seed <seed>
```

**Examples:**
```
python main.py --test --config default --model_name mppo_connect_v3 --seed 89
python main.py --test --config rw5 --model_name mppo_connect_rw5 --seed 42
```

**Testing Model Checkpoints (During Training)**
```
python main.py --test --config rw4 --model_name ../ckpt/mppo_connect_rw4/3 --seed 89
```

## Project Structure and Configuration

```
multi_robot_exploration/
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
