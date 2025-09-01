
import torch
import torch.nn as nn
from typing import Optional, Tuple

class ActorCriticCNNLSTM(nn.Module):
    def __init__(self, obs_space, num_outputs, lstm_hidden_size: int = 256, lstm_layers: int = 1):
        super().__init__()

        self.obs_space = obs_space
        h, w, c = obs_space.shape  # HWC

        # --- CNN encoder (includes MaxPool like your base) ---
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Infer flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat = self.conv(dummy)
            self.flattened_size = flat.size(1)

        # --- LSTM between CNN and heads ---
        self.lstm = nn.LSTM(
            input_size=self.flattened_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        # --- Heads (keep your 128->64 MLP) ---
        self.actor_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    # ---------- Utilities ----------
    def get_initial_state(self, batch_size: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero (h0, c0) with shapes [num_layers, B, hidden]."""
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def _to_channels_first(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B,H,W,C]/[B,T,H,W,C] -> channels-first."""
        c = self.obs_space.shape[-1]
        if x.ndim == 4:
            if x.shape[1] != c:  # [B,H,W,C] -> [B,C,H,W]
                x = x.permute(0, 3, 1, 2)
        elif x.ndim == 5:
            if x.shape[-1] == c:  # [B,T,H,W,C] -> [B,T,C,H,W]
                x = x.permute(0, 1, 4, 2, 3)
        return x

    def _encode_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """Run CNN; input either [B,C,H,W] or [B*T,C,H,W]; returns flat features."""
        return self.conv(x)

    # ---------- Forward ----------
    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dones: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            obs:
              - [B,C,H,W] or [B,H,W,C] (single step), or
              - [B,T,C,H,W] or [B,T,H,W,C] (sequence)
            state: optional (h0, c0) with shapes [num_layers, B, hidden]
            dones: optional boolean tensor:
                   - single step: [B]
                   - sequence:    [B,T]
                   Hidden state reset *after* each timestep where dones==True.

        Returns:
            logits, value, new_state
              - If sequence input: logits/value are [B,T,...]
              - If single step:    logits/value are [B,...]
        """
        x = self._to_channels_first(obs)

        if x.ndim == 4:
            # Single step
            B = x.size(0)
            feats = self._encode_cnn(x)             # [B, F]
            feats = feats.unsqueeze(1)              # [B, 1, F]
            if state is None:
                state = self.get_initial_state(B, device=feats.device)
            lstm_out, new_state = self.lstm(feats, state)  # [B,1,H]
            y = lstm_out.squeeze(1)                        # [B,H]
            logits = self.actor_head(y)                    # [B,A]
            value  = self.critic_head(y)                   # [B,1]

            # Optional: reset state positions where done=True (post-step)
            if dones is not None:
                dones = dones.to(dtype=torch.bool, device=y.device).view(-1)
                h, c = new_state
                h[:, dones] = 0
                c[:, dones] = 0
                new_state = (h, c)

            return logits, value, new_state

        elif x.ndim == 5:
            # Sequence
            B, T = x.size(0), x.size(1)
            x = x.reshape(B * T, x.size(2), x.size(3), x.size(4))  # [B*T,C,H,W]
            feats = self._encode_cnn(x)                            # [B*T, F]
            feats = feats.view(B, T, -1)                           # [B,T,F]

            if state is None:
                state = self.get_initial_state(B, device=feats.device)

            if dones is None:
                lstm_out, new_state = self.lstm(feats, state)      # [B,T,H]
            else:
                # Step through time so we can reset on dones
                h, c = state
                outs = []
                for t in range(T):
                    step_out, (h, c) = self.lstm(feats[:, t:t+1, :], (h, c))  # [B,1,H]
                    outs.append(step_out)
                    done_t = dones[:, t].to(dtype=torch.bool, device=feats.device)
                    # Reset AFTER producing t's output (so t is last step of old episode)
                    h[:, done_t] = 0
                    c[:, done_t] = 0
                lstm_out = torch.cat(outs, dim=1)  # [B,T,H]
                new_state = (h, c)

            flat = lstm_out.reshape(B * T, -1)           # [B*T,H]
            logits = self.actor_head(flat).view(B, T, -1)
            value  = self.critic_head(flat).view(B, T, 1)
            return logits, value, new_state

        else:
            raise ValueError(f"obs must be 4D or 5D, got {x.ndim}D")


# --- Backwards-compatible alias for rl_wrapper ---
# rl_wrapper expects a symbol named ActorCriticCNNModel to exist.
class ActorCriticCNNModel(ActorCriticCNNLSTM):
    pass
