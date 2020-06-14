import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.networks.shared.general import LinearLayer

from utils.config_handler import cf
NUM_RNN_UNITS = cf.val("NUM_RNN_UNITS")
OBS_EMBED_SIZE = cf.val("OBS_EMBED_SIZE")
CNN_SIZE_1 = cf.val("CNN_SIZE_1")
CNN_SIZE_2 = cf.val("CNN_SIZE_2")
CNN_SIZE_3 = cf.val("CNN_SIZE_3")
APPLY_RELU_TO_OBS_EMBED = cf.val("APPLY_RELU_TO_OBS_EMBED")


class CNN_GRU_Network(nn.Module):
    def __init__(self, input_size, action_space_size):
        super(CNN_GRU_Network, self).__init__()
        next_input_size = input_size
        if OBS_EMBED_SIZE > 0:
            self.obs_emb = LinearLayer(next_input_size, OBS_EMBED_SIZE)
            next_input_size = OBS_EMBED_SIZE

        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=CNN_SIZE_1, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=CNN_SIZE_1, out_channels=CNN_SIZE_2, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=CNN_SIZE_2, out_channels=CNN_SIZE_3, kernel_size=(2, 2)),
            nn.ReLU()
        )
        next_input_size += CNN_SIZE_3

        self.num_rnn_units = NUM_RNN_UNITS
        self.rnn = nn.GRUCell(next_input_size, self.num_rnn_units)

    def forward(self, obs, old_state):
        obs_vec = obs[0]
        obs_image = obs[1]

        image_ten = torch.FloatTensor(obs_image).unsqueeze(0)

        x = torch.transpose(torch.transpose(image_ten, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)


        tens = torch.FloatTensor(obs_vec).unsqueeze(0)
        if OBS_EMBED_SIZE > 0:
            tens = self.obs_emb(tens)
            if APPLY_RELU_TO_OBS_EMBED:
                tens = F.relu(tens)

        tens = torch.cat([x, tens], 1)

        new_state = self.rnn(tens, old_state)
        return new_state, new_state

    def init_state(self):
        return torch.zeros(1, self.num_rnn_units)

    def detach_from_history(self, state):
        return state.detach()
