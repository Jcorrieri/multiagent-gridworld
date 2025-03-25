from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.policy.policy import PolicySpec


def build_config(env_config: dict) -> Algorithm:
    config = (
        PPOConfig()
        .environment(
            env="grid_world",
            env_config=env_config,
            render_env=False,
            is_atari=False,
        )
        .framework("torch")
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .training(
            model={
                "custom_model": "shared_cnn",
                "vf_share_layers": False
            },
            gamma=0.9, # match MATLAB
            lr=1e-3, # match MATLAB
            grad_clip=1.0, # match MATLAB
            train_batch_size_per_learner=500, # match MATLAB
            minibatch_size=200, # match MATLAB
        )
        .api_stack(
            enable_env_runner_and_connector_v2=False,
            enable_rl_module_and_learner=False,
        )
    )

    print("Multi-Agent Config:", config.is_multi_agent)

    return config.build_algo()

def parse_optimizer(parser):
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, default='models/saved/mppo')
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--obs_mat_path', type=str, default='env/obstacle_mats/mat1')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--size', type=int, default=12)
    parser.add_argument('--num_agents', type=int, default=3)
    parser.add_argument('--cr', type=int, default=3)
