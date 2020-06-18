# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.networks.shared.general import LinearLayer, ResidualLayer, LayerNorm

from utils.config_handler import cf
OBS_EMBED_SIZE = cf.val("OBS_EMBED_SIZE")
APPLY_RELU_TO_OBS_EMBED = cf.val("APPLY_RELU_TO_OBS_EMBED")
NUM_FF_LAYERS = cf.val("NUM_FF_LAYERS")
if NUM_FF_LAYERS > 0:
    FF_LAYER_SIZE = cf.val("FF_LAYER_SIZE")
    RESFF_APPLY_LAYER_NORM = cf.val("RESFF_APPLY_LAYER_NORM")
    RESFF_APPLY_RELU_AT_LAYER_END = cf.val("RESFF_APPLY_RELU_AT_LAYER_END")


class ResFF_Layer(nn.Module):
    def __init__(self, inter_layer_size, intra_layer_size):
        super(ResFF_Layer, self).__init__()
        self.feedforward = LinearLayer(inter_layer_size, intra_layer_size, std)
        self.feedforward_residual = ResidualLayer(intra_layer_size, inter_layer_size)
        if RESFF_APPLY_LAYER_NORM:
            self.feedforward_layer_norm = LayerNorm(inter_layer_size)

    def forward(self, input):
        tens = self.feedforward(input)
        tens = F.relu(tens)
        tens = self.feedforward_residual(tens, input)
        if RESFF_APPLY_LAYER_NORM:
            tens = self.feedforward_layer_norm(tens)
        if RESFF_APPLY_RELU_AT_LAYER_END:
            tens = F.relu(tens)
        return tens


class ResFF_Network(nn.Module):
    def __init__(self, input_size, action_space_size):
        super(ResFF_Network, self).__init__()
        next_input_size = input_size
        if OBS_EMBED_SIZE > 0:
            self.obs_proj = LinearLayer(next_input_size, OBS_EMBED_SIZE)
            next_input_size = OBS_EMBED_SIZE
        self.ff_layers = None
        if NUM_FF_LAYERS > 0:
            self.ff_layers = nn.ModuleList()
            for l in range(NUM_FF_LAYERS):
                self.ff_layers.append(ResFF_Layer(next_input_size, FF_LAYER_SIZE))
                #next_input_size = FF_LAYER_SIZE

    def forward(self, observation, old_state):
        tens = torch.FloatTensor(observation).unsqueeze(0)
        if OBS_EMBED_SIZE > 0:
            tens = self.obs_proj(tens)
        if APPLY_RELU_TO_OBS_EMBED:
            tens = F.relu(tens)
        if self.ff_layers != None:
            for layer in self.ff_layers:
                tens = layer(tens)
        return tens, None

    def init_state(self):
        return None

    def detach_from_history(self, state):
        return None
