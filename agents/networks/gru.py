# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.networks.shared.general import LinearLayer, ActorCriticLayers

from utils.config_handler import cf
NUM_RNN_UNITS = cf.val("NUM_RNN_UNITS")
AC_HIDDEN_LAYER_SIZE = cf.val("AC_HIDDEN_LAYER_SIZE")
OBS_EMBED_SIZE = cf.val("OBS_EMBED_SIZE")
APPLY_RELU_TO_OBS_EMBED = cf.val("APPLY_RELU_TO_OBS_EMBED")


class GRU_Network(nn.Module):
    def __init__(self, input_size, action_space_size):
        super(GRU_Network, self).__init__()
        next_input_size = input_size
        if OBS_EMBED_SIZE > 0:
            self.obs_emb = LinearLayer(next_input_size, OBS_EMBED_SIZE)
            next_input_size = OBS_EMBED_SIZE
        self.num_rnn_units = NUM_RNN_UNITS
        self.rnn = nn.GRUCell(next_input_size, self.num_rnn_units)
        self.actor_critic_layers = ActorCriticLayers(self.num_rnn_units, 2, AC_HIDDEN_LAYER_SIZE, action_space_size)

    def forward(self, obs, old_state):
        tens = torch.FloatTensor(obs).unsqueeze(0)
        if OBS_EMBED_SIZE > 0:
            tens = self.obs_emb(tens)
            if APPLY_RELU_TO_OBS_EMBED:
                tens = F.relu(tens)
        new_state = self.rnn(tens, old_state)
        value_est, policy = self.actor_critic_layers(new_state)
        return value_est, policy, new_state

    def init_state(self):
        return torch.zeros(1, self.num_rnn_units)

    def detach_from_history(self, state):
        return state.detach()
