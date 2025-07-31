# Multi-Robot Exploration with RLlib

This is a Deep Reinforcement Learning (DRL) framework for coordinating a team of robots to explore an unknown grid-world environment while maintaining a connected communication network. The proposed system adapts a multi-agent actor-critic algorithm (Multi-Agent PPO) using Ray RLlib’s legacy API and PettingZoo (Gymnasium) for environment modeling. Inspired by a MATLAB-based area coverage example, this project extends the task to include communication constraints by modifying the reward function. Empirical results demonstrate that the learned policy achieves moderate to high success rates in preserving network connectivity during exploration, and shows potential for generalization to larger teams.

## Installation (Python v3.10)

**Install Pytorch with CUDA support (Required)**
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_ver}
```

**Install Dependencies (Python version 3.10 required for pre-trained models -- I use 3.10.11)**
```
pip install ray[tune]==2.44.1 PettingZoo>=1.22.3 dm_tree scipy pygame matplotlib lz4 pyyaml
```

## Project Structure

```YAML
/
│
├── config/                                         # YAML project config files
│   ├── default.txt
│   └── ...
│
├── environment/
│   ├── envs/                              
│   │   ├── gridworld.py
│   │   └── ...
│   ├── obstacle_mats/                              # custom obstacle maps (origin at top-left)
│   │   ├── testing/
│   │   │   ├── mat0
│   │   │   └── ...
│   │   └── training/
│   │       ├── mat0
│   │       └── ...
│   └── rewards.py                              
│
├── experiments/                                    # training and testing files split by scenario
│   ├── base-station/                               
│   │   └── v0/                                     
│   │       ├── ckpt/                               # model checkpoint(s) during training
│   │       │   ├── 0/                              # checkpoint 0, 1, ..., n
│   │       │   │   └── <rllib_algorithm_files>
│   │       │   └── ...
│   │       ├── saved/                              # the final model after training is finished
│   │       │   └── <rllib_algorithm_files>
│   │       ├── test-results/               
│   │       │   └── results.csv
│   │       ├── train-metrics/
│   │       │   └── metrics_plot.png
│   │       └── config.txt                          # copy of config/<config.txt> used for training
│   ├── baseline/
│   │   ├── v0/  
│   │   │   └── ...
│   │   └── ...
│   └── default-env/
│       ├── v0/  
│       │   └── ...
│       └── ...
│
├── models/                                         # neural network architectures
│   ├── arch/
│   │   ├── cnn_2conv3linear.py
│   │   └── ...                        
│   └── rl_wrapper.py                               # wrapper for Ray RLlib
│
├── main.py                                         # main entry point for training and testing
├── test.py                                         
├── train.py
└── utils.py                                        # handles arguments, environments, and metrics
```

## Usage:

Everything is configured through a central configuration file located under ```config/```

```yaml
environment:
  env_name: gridworld                               # gridworld or baseline
  base_station: False                               # include fixed base station
  fov: 4                                  
  max_steps: 1000                          
  size: 25
  num_agents: 5
  cr: 10                                            # communication range between agents

reward_scheme:
  module: components
  params:
    new_tile: 0.5
    total_coverage_bonus: 0.5
    total_teammate_bonus: 1.0
    missing_teammate_penalty: -0.8
    termination_bonus: 100

training:
  module_file: cnn_2conv3linear.py
  num_episodes: 5000
  target_reward: 10000
  gamma: 0.99
  lr: 0.0001
  grad_clip: 1.0
  train_batch_size: 8192
  num_passes: 5
  minibatch_size: 1024
  l2_regularization: 0.0001
  lambda_: 0.95
  entropy_coeff: [[0, 0.01]]
  clip_param: 0.2
  vf_clip_param: 10.0

testing:
  num_episodes_per_map: 10                          # num_episodes * 50 maps  
  seed: 42
  explore: True                                     # recommended value: True
  render: True
  model_path: default-env/v0
  checkpoint: -1                                    # set > -1 to restore from a checkpoint (model_path/ckpt/<number>)
```

### Training Models

Run ```python main.py```

- Select a neural network architecture from ```models/arch/``` using the ```module_file``` parameter.
  - You can add your own architecture(s) as long as it mimics the ActorCriticCNNModel class.
- You can select how many episodes to train for using ```num_episodes```.
- A copy of the configuration when training will be saved at ```...model_path/config.txt```. 
- Model checkpoints will be saved every 200 iterations at ```...model_path/ckpt/i/```, where ```i``` is the checkpoint number.
- After training, the model will be saved at ```...model_path/saved/``` and the training metrics at ```...model_path/train-metrics/```.

### Testing Models

Run ```python main.py --test```

- Select the model to test by specifying its relative path (i.e. ```default-env/v1```)
- Specify a checkpoint number greater than -1 to test a checkpoint from ```...model_path/ckpt/<i>```.
- Set ```render: True``` to display the environments during testing (will slow down evaluation)
- A csv file with various results will be saved at ```...model_path/test-results/```

### Additional Notes

- Use ```utils.gen_train_test_split()``` to generate 100 new maps for training and testing.
  - Tune obstacle density by changing the corresponding parameter in ```utils.generate_obstacles(...)```.
- Add custom configuration files under ```config/``` and use ```--config <file_name>``` to specify during training and testing.
  - ```python main.py --config <custom_config>```
  - ```python main.py --test --config <custom_config>```