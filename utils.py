from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec


def build_config(env_config: dict) -> Algorithm:
    config = (
        PPOConfig()
        .environment("grid_world", env_config=env_config)
        .framework("torch")
        .resources(num_cpus_per_worker=1)
        .rollouts(num_rollout_workers=0, rollout_fragment_length=500)  # Use 0 for Windows/dev, 500 = horizon
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"]
        )
        .training(
            model={
                "custom_model": "shared_cnn",
                "vf_share_layers": False,
            },
            gamma=0.9, # match MATLAB
            lr=1e-3, # match MATLAB
            grad_clip=1.0, # match MATLAB
            train_batch_size=500, # match MATLAB
            sgd_minibatch_size=200, # match MATLAB
            num_sgd_iter=10, # match MATLAB
        )
    )

    return config.build()

def parse_optimizer(parser):
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, default='models/saved/mppo')
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--obs_mat_path', type=str, default='env/obstacle_mats/mat1')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--size', type=int, default=12)
    parser.add_argument('--num_agents', type=int, default=3)
    parser.add_argument('--cr', type=int, default=3)
