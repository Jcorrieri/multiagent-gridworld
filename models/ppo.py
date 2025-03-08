from argparse import Namespace

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from models.Wrappers import TQDMCallback
from models.gnn_feat_extractor import GNNFeatureExtractor


class CustomPPO(nn.Module):
    def __init__(self, args: Namespace, default: bool = False):
        super(CustomPPO, self).__init__()
        self.args = args
        self.model_name = args.model_name
        if args.test:
            self.model = PPO.load("models/saved/"+self.model_name)
        elif default:
            self.model = PPO(
                "MultiInputPolicy",
                args.env,
                verbose=0,
                device=args.device,
                tensorboard_log = "models/saved/logs/"+self.model_name+"/",
            )

        else:
            policy_kwargs = {
                "features_extractor_class": GNNFeatureExtractor,
            }
            self.model = PPO(
                "MultiInputPolicy",
                args.env,
                verbose=0,
                policy_kwargs=policy_kwargs,
                device=args.device,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                tensorboard_log="models/saved/logs/"+self.model_name+"/",
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

        checkpoint_callback = CheckpointCallback(
            save_freq=500000,
            save_path="models/saved/ckpt/",
            name_prefix=self.model_name+"_ckpt_",
        )

        callback_list = CallbackList([checkpoint_callback, TQDMCallback(total_timesteps=total_timesteps)])

        self.model.learn(total_timesteps=total_timesteps, callback=callback_list)
        self.model.save(self.model_name)