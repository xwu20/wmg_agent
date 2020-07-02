# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import numpy as np

from agents.networks.shared.transformer import Transformer
from agents.networks.shared.general import LinearLayer
from agents.networks.shared.general import ActorCriticLayers

from utils.config_handler import cf
TFM_ATTENTION_HEAD_SIZE = cf.val("TFM_ATTENTION_HEAD_SIZE")
TFM_NUM_ATTENTION_HEADS = cf.val("TFM_NUM_ATTENTION_HEADS")
TFM_NUM_LAYERS = cf.val("TFM_NUM_LAYERS")
TFM_HIDDEN_SIZE = cf.val("TFM_HIDDEN_SIZE")
TFM_INITIALIZER_RANGE = cf.val("TFM_INITIALIZER_RANGE")
AC_HIDDEN_LAYER_SIZE = cf.val("AC_HIDDEN_LAYER_SIZE")
TFM_OUTPUT_ALL_NODES = cf.val("TFM_OUTPUT_ALL_NODES")
WMG_MAX_MEMS = cf.val("WMG_MAX_MEMS")
WMG_HIST_MEMS = cf.val("WMG_HIST_MEMS")
if not WMG_HIST_MEMS:
    WMG_MEM_SIZE = cf.val("WMG_MEM_SIZE")


class MemoryBank(object):
    def __init__(self, mem_len):
        self.mem_len = mem_len
        self.memories = [torch.tensor(np.zeros(self.mem_len, np.float32)) for n in range(WMG_MAX_MEMS)]
        self.idx = 0  # Where to store the next memory, overwriting the oldest.

    def get_memory(self, i):
        assert (i >= 0) and (i < WMG_MAX_MEMS)
        return self.memories[(self.idx - i + WMG_MAX_MEMS) % WMG_MAX_MEMS]

    def add_memory(self, memory):
        self.memories[self.idx] = memory
        self.idx = (self.idx + 1) % WMG_MAX_MEMS

    def copy(self):
        g = MemoryBank(self.mem_len)
        for i in range(WMG_MAX_MEMS):
            g.memories[i] = self.memories[i]
        g.idx = self.idx
        return g

    def detach_from_history(self):
        for i in range(WMG_MAX_MEMS):
            self.memories[i] = self.memories[i].detach()


class WMG_Network(nn.Module):
    def __init__(self, observation_space_size, action_space_size, factored_observations = True):
        super(WMG_Network, self).__init__()
        self.factored_observations = factored_observations
        self.node_vec_size = TFM_NUM_ATTENTION_HEADS * TFM_ATTENTION_HEAD_SIZE
        if self.factored_observations:
            self.global_vec_size = observation_space_size[0]
            self.factor_vec_size = observation_space_size[1]
        else:
            self.global_vec_size = observation_space_size
        if WMG_HIST_MEMS:
            assert not self.factored_observations
            self.memory_vec_size = self.global_vec_size
        else:
            self.memory_vec_size = WMG_MEM_SIZE
        self.age_vec_size = WMG_MAX_MEMS
        self.global_embed_layer = LinearLayer(self.global_vec_size, self.node_vec_size)
        if self.factored_observations:
            self.factor_embed_layer = LinearLayer(self.factor_vec_size, self.node_vec_size)
        self.memory_embed_layer = LinearLayer(self.memory_vec_size + self.age_vec_size, self.node_vec_size)

        # Prepare the age vectors.
        self.age = []
        for i in range(WMG_MAX_MEMS):
            pos = np.zeros(self.age_vec_size, np.float32)
            pos[i] = 1.
            self.age.append(torch.tensor(pos))

        self.tfm = Transformer(TFM_NUM_ATTENTION_HEADS, TFM_ATTENTION_HEAD_SIZE,
                               TFM_NUM_LAYERS, TFM_HIDDEN_SIZE)
        self.actor_critic_layers = ActorCriticLayers(self.node_vec_size, 2, AC_HIDDEN_LAYER_SIZE, action_space_size)
        if not WMG_HIST_MEMS:
            self.memory_def_layer = LinearLayer(self.node_vec_size, self.memory_vec_size)

    def forward(self, observation, old_state):
        # Gather the node embeddings.
        node_vec_list = []
        if self.factored_observations:
            node_vec_list.append(self.global_embed_layer(torch.tensor(np.float32(observation[0]))))
            for factor_vec in observation[1:]:
                node_vec_list.append(self.factor_embed_layer(torch.tensor(np.float32(factor_vec))))
        else:
            global_obs_tens = torch.tensor(np.float32(observation))
            node_vec_list.append(self.global_embed_layer(global_obs_tens))
        for i in range(WMG_MAX_MEMS):
            node_vec_list.append(self.memory_embed_layer(torch.cat((old_state.get_memory(i), self.age[i]))))

        # Reshape the tensor.
        graph_input = torch.cat(node_vec_list).view(1, len(node_vec_list), self.node_vec_size)

        # Run the data through the transformer.
        graph_output = self.tfm(graph_input)

        # Pass only the global (first) node's output to the subsequent layers.
        tfm_output = graph_output[:,0,:]  # From the global node only.
        value_est, policy = self.actor_critic_layers(tfm_output)
        if WMG_HIST_MEMS:
            new_memory = torch.tanh(global_obs_tens)
        else:
            new_memory = torch.tanh(self.memory_def_layer(tfm_output))[0]

        # Update the state.
        new_state = old_state.copy()
        new_state.add_memory(new_memory)

        return value_est, policy, new_state

    def init_state(self):
        state = MemoryBank(self.memory_vec_size)
        return state

    def detach_from_history(self, state):
        state.detach_from_history()
        return state
