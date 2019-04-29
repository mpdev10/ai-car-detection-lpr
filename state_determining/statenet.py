import torch.nn as nn


class StateNet(nn.Module):
    def __init__(self, frame_num, num_states):
        super(StateNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(4 * frame_num, 256),
            nn.ReLU(),
            nn.Linear(256, num_states,
                      nn.Sigmoid())
        )

    def forward(self, x):
        x = self.model(x)
        return x
