from argparse import Namespace

import torch.nn as nn
from stable_baselines3 import PPO

from models.Wrappers import TQDMCallback
from models.gnnppo import GNNFeatureExtractor


class CustomPPO(nn.Module):
    def __init__(self, args: Namespace, default: bool = False):
        super(CustomPPO, self).__init__()
        self.args = args
        self.filename = "models/saved/Custom_PPO" if args.model_path is None else args.model_path
        if args.test:
            self.model = PPO.load(self.filename)
        elif default:
            self.model = PPO(
                "MultiInputPolicy",
                args.env,
                verbose=0,
                device=args.device,
                tensorboard_log = "models/saved/logs/"
            )

        else:
            policy_kwargs = dict(
                features_extractor_class=GNNFeatureExtractor,
                #features_extractor_kwargs=dict(hidden_dim=128, out_dim=64),
            )
            self.model = PPO(
                "MultiInputPolicy",
                args.env,
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=args.device,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                tensorboard_log="models/saved/logs/",
                gamma=0.99,
                ent_coef=0.03,
                gae_lambda=0.95,
                clip_range=0.2
            )

    def forward(self, x):
        action, _states = self.model.predict(x)
        return action

    def learn(self):
        total_timesteps = 10000000
        self.model.learn(total_timesteps=total_timesteps, callback=TQDMCallback(total_timesteps))
        self.model.save(self.filename)