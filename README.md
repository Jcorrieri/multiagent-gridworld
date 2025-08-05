# Multi-Robot Exploration with RLlib

This is a Deep Reinforcement Learning (DRL) framework for coordinating a team of robots to explore an unknown grid-world environment while maintaining a connected communication network. The proposed system adapts a multi-agent actor-critic algorithm (Multi-Agent PPO) using Ray RLlib’s legacy API and PettingZoo (Gymnasium) for environment modeling. Inspired by a MATLAB-based area coverage example, this project extends the task to include communication constraints by modifying the reward function.

## Installation (Python v3.10)

**Install Pytorch with CUDA support (Required)**
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_ver}
```

**Install Dependencies (Python version 3.10 required for pre-trained models -- I use 3.10.11)**
```
pip install ray[tune]==2.44.1 PettingZoo>=1.22.3 dm_tree scipy pygame matplotlib lz4 pyyaml
```

## Usage:

Everything is configured through a central configuration file located at ```config/default```.

### Environment

Specify the environment parameters for training and testing. The map size must remain consistent between training and testing, but all other parameters can be adjusted accordingly.
- reward_scheme: Change between any reward scheme defined in ```utils.make_reward_scheme```.

### Training

- Select a neural network architecture from ```models/arch/``` using the ```module_file``` parameter.
- Note that, when specifying an entropy_coeff schedule, you should multiply each timestep value by ```num_agents``` to get an accurate schedule during training.
  - The schedule uses a format of [[timestep, value], ..., [timestep, value]].
- Model checkpoints, results, and a copy of the config file will be saved under ```experiment/<env_name>/<verion>```.

When everything is set as you prefer in the config file, run ```python main.py```. You can view live metrics by running ```tensorboard --logdir=~\ray_results``` and selecting the latest run.

### Testing

- Select the model to test (i.e. ```v1```)
- Specify a checkpoint number greater than -1 to test a checkpoint from ```...model_path/ckpt/<i>```, otherwise use the latest saved model.
- Set ```explore: True``` to enable some stochasticity.
- A csv file with various results will be saved at ```...model_path/test-results/```

Run ```python main.py --test``` to begin testing.

### Testing the Baseline

For the baseline, I implemented the following paper: [Multi-robot exploration under the constraints of wireless networking](https://www.sciencedirect.com/science/article/pii/S0967066106001547#bib16)
- To test the baseline, change ```environment:env_name``` to "baseline" in the config file. This does not use a trained model.

### Additional Notes

- Use ```utils.gen_train_test_split()``` to generate 100 new maps for training and testing.
  - Tune obstacle densities by modifying the function.
- Define new reward schemes and modify existing ones in ```environment_rewards.py```. Add new schemes to ```utils.make_reward_scheme``` to use them in your config.
- Add custom configuration files under ```config/``` and use ```--config <file_name>``` to specify during training and testing.
  - ```python main.py --config <custom_config>```
  - ```python main.py --test --config <custom_config>```
- See the [Ray RLlib documentation](https://docs.ray.io/en/latest/rllib/index.html) for ray-specific info. I use the old stack (not RLModule).

## Project Structure

```YAML
/
├── config/                                         # YAML project config files
│   └── default.txt
│
├── environment/
│   ├── envs/                              
│   │   ├── gridworld.py
│   │   └── baseline.py
│   ├── obstacle_mats/                              # custom obstacle maps (origin at top-left)
│   │   ├── testing/
│   │   │   ├── mat0
│   │   │   └── ...
│   │   └── training/
│   │       ├── mat0
│   │       └── ...
│   └── rewards.py                                  # create and customize reward functions                           
│
├── experiments/                                    # training and testing files split by scenario
│   ├── gridworld/                               
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
│   └── baseline/
│       ├── v0/  
│       │   └── ...
│       └── ...
│
├── models/                                         # neural network architectures
│   ├── arch/
│   │   ├── cnn_2conv2linear.py
│   │   └── ...                        
│   └── rl_wrapper.py                               # wrapper for Ray RLlib
│
├── main.py                                         # main entry point for training and testing
├── test.py                                         
├── train.py
└── utils.py                                        # handles arguments, environments, and metrics
```