import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q-network for TactMAE active touch exploration.
    Input: cat(action_emb, current_latent, first_latent)
    Output: Q-values for each of num_actions touches.
    """

    def __init__(self, args):
        super().__init__()
        na = args.num_actions
        ld = args.latent_dim  # 192

        # embed the binary action mask (which touches have been used)
        self.action_model = nn.Sequential(
            nn.Linear(na, 200), nn.ReLU(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, ld),
        )

        # MLP: cat(action_emb[ld], current_latent[ld], first_latent[ld]) → num_actions Q-values
        hidden = args.hidden_dim
        n_layers = args.layers
        sizes = [ld * 3] + [hidden] * (n_layers - 1) + [na]
        layers = []
        for i in range(n_layers):
            if i < n_layers - 1:
                layers.append(nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()))
            else:
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.model = nn.Sequential(*layers)

    def forward(self, obs, next=False):
        if next:
            action_emb = self.action_model(obs['mask_n'].float().cuda())
            latent = obs['latent_n'].float().cuda()
        else:
            action_emb = self.action_model(obs['mask'].float().cuda())
            latent = obs['latent'].float().cuda()
        first_latent = obs['first_latent'].float().cuda()
        x = torch.cat([action_emb, latent, first_latent], dim=-1)
        return self.model(x)
