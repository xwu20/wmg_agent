# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.networks.shared.transformer import Transformer
from agents.networks.shared.general import LinearLayer
from agents.networks.shared.general import ActorCriticLayers

from utils.config_handler import cf
REFACTORED = cf.val("REFACTORED")
TFM_ATTENTION_HEAD_SIZE = cf.val("TFM_ATTENTION_HEAD_SIZE")
TFM_NUM_ATTENTION_HEADS = cf.val("TFM_NUM_ATTENTION_HEADS")
TFM_NUM_LAYERS = cf.val("TFM_NUM_LAYERS")
TFM_HIDDEN_SIZE = cf.val("TFM_HIDDEN_SIZE")
AC_HIDDEN_LAYER_SIZE = cf.val("AC_HIDDEN_LAYER_SIZE")
WMG_MAX_OBS = cf.val("WMG_MAX_OBS")
WMG_MAX_MEMOS = cf.val("WMG_MAX_MEMOS")

# This implementation does not support both Memos and past observations.
assert (WMG_MAX_MEMOS == 0) or (WMG_MAX_OBS == 0)
MAX_STATE_VECTORS = WMG_MAX_MEMOS + WMG_MAX_OBS
if WMG_MAX_MEMOS:
    WMG_MEMO_SIZE = cf.val("WMG_MEMO_SIZE")


class StateMatrix(object):
    ''' Used to store either Factors (from past observations) or Memos. '''
    def __init__(self, vector_len, max_vectors):
        self.vector_len = vector_len
        self.max_vectors = max_vectors
        self.num_vectors = 0  # Capped at self.max_vectors.
        self.vectors = [torch.tensor(np.zeros(self.vector_len, np.float32)) for n in range(self.max_vectors)]
        self.next_index = 0  # Where to store the next vector, overwriting the oldest.

    def get_vector(self, i):
        assert (i >= 0) and (i < self.max_vectors)
        if REFACTORED:
            i += 1
        return self.vectors[(self.next_index - i + self.max_vectors) % self.max_vectors]

    def add_vector(self, vector):
        self.vectors[self.next_index] = vector
        self.next_index = (self.next_index + 1) % self.max_vectors
        if self.num_vectors < self.max_vectors:
            self.num_vectors += 1

    def clone(self):
        g = StateMatrix(self.vector_len, MAX_STATE_VECTORS)
        for i in range(self.max_vectors):
            g.vectors[i] = self.vectors[i].clone()
        g.next_index = self.next_index
        g.num_vectors = self.num_vectors
        return g

    def detach_from_history(self):
        for i in range(self.max_vectors):
            self.vectors[i] = self.vectors[i].detach()


class WMG_Network(nn.Module):
    def __init__(self, observation_space_size, action_space_size, factored_observations):
        super(WMG_Network, self).__init__()
        self.factored_observations = factored_observations
        self.tfm_vec_size = TFM_NUM_ATTENTION_HEADS * TFM_ATTENTION_HEAD_SIZE

        if REFACTORED:
            if self.factored_observations:
                self.factor_sizes = observation_space_size.entity_type_sizes
                self.embedding_layers = nn.ModuleList()
                for factor_size in self.factor_sizes:
                    self.embedding_layers.append(LinearLayer(factor_size, self.tfm_vec_size))
            else:
                self.core_embed_layer = LinearLayer(observation_space_size, self.tfm_vec_size)
        else:
            if self.factored_observations:
                self.global_vec_size = observation_space_size[0]
                self.factor_vec_size = observation_space_size[1]
            else:
                self.global_vec_size = observation_space_size

        if WMG_MAX_OBS:
            assert not self.factored_observations
            self.state_vector_len = observation_space_size
        else:
            self.state_vector_len = WMG_MEMO_SIZE

        if REFACTORED:
            if MAX_STATE_VECTORS > 0:
                self.concept_embed_layer = LinearLayer(WMG_MEMO_SIZE, self.tfm_vec_size)
        else:
            self.age_vec_len = MAX_STATE_VECTORS
            self.global_embed_layer = LinearLayer(self.global_vec_size, self.tfm_vec_size)
            if self.factored_observations:
                self.factor_embed_layer = LinearLayer(self.factor_vec_size, self.tfm_vec_size)
            self.memory_embed_layer = LinearLayer(self.state_vector_len + self.age_vec_len, self.tfm_vec_size)

        # Prepare the age vectors.
        self.age = []
        if REFACTORED:
            # self.age contains a total of MAX_STATE_VECTORS+2 vectors, each of size MAX_STATE_VECTORS+2.
            # self.age[0] is for the current time step.
            # self.age[MAX_STATE_VECTORS] is for the oldest time step in the history.
            # self.age[MAX_STATE_VECTORS+1] is for out-of-range ages.
            # One additional element is copied into any age vector to hold the raw age value itself.
            # Each vector is of size MAX_STATE_VECTORS+2, for a 1-hot vector of size MAX_STATE_VECTORS+1, and the raw age itself.
            self.max_age_index = MAX_STATE_VECTORS + 1
            age_vector_size = MAX_STATE_VECTORS + 2
            for i in range(self.max_age_index):
                pos = np.zeros(age_vector_size, np.float32)
                pos[i] = 1.
                self.age.append(pos)
            self.age.append(np.zeros(age_vector_size, np.float32))
            self.age_embed_layer = LinearLayer(age_vector_size, self.tfm_vec_size)
        else:
            for i in range(MAX_STATE_VECTORS):
                pos = np.zeros(self.age_vec_len, np.float32)
                pos[i] = 1.
                self.age.append(torch.tensor(pos))

        self.tfm = Transformer(TFM_NUM_ATTENTION_HEADS, TFM_ATTENTION_HEAD_SIZE,
                               TFM_NUM_LAYERS, TFM_HIDDEN_SIZE)

        if not REFACTORED:
            self.actor_critic_layers = ActorCriticLayers(self.tfm_vec_size, 2, AC_HIDDEN_LAYER_SIZE, action_space_size)

        if REFACTORED:
            if MAX_STATE_VECTORS > 0:
                self.concept_creation_layer = LinearLayer(self.tfm_vec_size, WMG_MEMO_SIZE)
        else:
            if WMG_MAX_MEMOS:
                self.memory_def_layer = LinearLayer(self.tfm_vec_size, self.state_vector_len)

        if REFACTORED:
            self.output_size = AC_HIDDEN_LAYER_SIZE
            self.output_layer = LinearLayer(self.tfm_vec_size, self.output_size)

    def add_age_embedding(self, vector_in, age):
        age_vec = self.age[min(age, self.max_age_index)]
        age_vec[-1] = age / MAX_STATE_VECTORS  # Include the raw age to convey order.
        age_embedding = self.age_embed_layer(torch.tensor(age_vec))
        return vector_in + age_embedding

    def forward(self, observation, old_matrix):
        # Embed the core vector, and any factor vectors.
        embedded_vec_list = []
        if self.factored_observations:
            if REFACTORED:
                for entity in observation.entities:
                    embedded_node_vec = self.embedding_layers[entity.type](torch.tensor(entity.data))
                    if entity.age is not None:
                        embedded_node_vec = self.add_age_embedding(embedded_node_vec, factor_age)
                    embedded_vec_list.append(embedded_node_vec)
            else:
                embedded_vec_list.append(self.global_embed_layer(torch.tensor(np.float32(observation[0]))))
                for factor_vec in observation[1:]:
                    embedded_vec_list.append(self.factor_embed_layer(torch.tensor(np.float32(factor_vec))))
        else:
            observations_tens = torch.tensor(np.float32(observation))
            embedded_vec_list.append(self.global_embed_layer(observations_tens))

        # Embed any state vectors.
        if REFACTORED:
            for i in range(old_matrix.num_vectors):
                embedded_vector = self.add_age_embedding(self.concept_embed_layer(old_matrix.get_vector(i)), i)
                embedded_vec_list.append(embedded_vector)
        else:
            for i in range(MAX_STATE_VECTORS):
                embedded_vector = self.memory_embed_layer(torch.cat((old_matrix.get_vector(i), self.age[i])))
                embedded_vec_list.append(embedded_vector)

        # Reshape the tensor for the Transformer to handle.
        graph_input = torch.cat(embedded_vec_list).view(1, len(embedded_vec_list), self.tfm_vec_size)

        # Run the data through the Transformer.
        graph_output = self.tfm(graph_input)

        # Pass only the core (first) column's output to the subsequent layers.
        tfm_output = graph_output[:,0,:]

        # Update any state.
        if MAX_STATE_VECTORS > 0:
            if WMG_MAX_MEMOS:
                # Create a new memo.
                if REFACTORED:
                    new_vector = torch.tanh(self.concept_creation_layer(tfm_output))[0]
                else:
                    new_vector = torch.tanh(self.memory_def_layer(tfm_output))[0]
            else:
                # Create a new observation factor.
                new_vector = torch.tanh(observations_tens)
            new_matrix = old_matrix.clone()
            new_matrix.add_vector(new_vector)
        else:
            new_matrix = None

        # Prepare the output.
        if REFACTORED:
            output = F.relu(self.output_layer(tfm_output))
            ret = output, new_matrix
        else:
            value_est, policy = self.actor_critic_layers(tfm_output)
            ret = value_est, policy, new_matrix

        return ret

    def init_state(self):
        if MAX_STATE_VECTORS > 0:
            return StateMatrix(self.state_vector_len, MAX_STATE_VECTORS)
        else:
            return None

    def detach_from_history(self, matrix):
        if matrix is not None:
            matrix.detach_from_history()
        return matrix
