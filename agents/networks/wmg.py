# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import numpy as np

from agents.networks.shared.transformer import Transformer
from agents.networks.shared.general import LinearLayer
from agents.networks.shared.general import ActorCriticLayers
from agents.networks.shared.general import SharedActorCriticLayers
from utils.graph import Graph

from utils.config_handler import cf
V2 = cf.val("V2")
TFM_ATTENTION_HEAD_SIZE = cf.val("TFM_ATTENTION_HEAD_SIZE")
TFM_NUM_ATTENTION_HEADS = cf.val("TFM_NUM_ATTENTION_HEADS")
TFM_NUM_LAYERS = cf.val("TFM_NUM_LAYERS")
TFM_HIDDEN_SIZE = cf.val("TFM_HIDDEN_SIZE")
AC_HIDDEN_LAYER_SIZE = cf.val("AC_HIDDEN_LAYER_SIZE")
WMG_MAX_OBS = cf.val("WMG_MAX_OBS")
WMG_MAX_MEMOS = cf.val("WMG_MAX_MEMOS")

# Set WMG_MAX_OBS > 0 for attention over past observations, stored in a StateMatrix.
# Set WMG_MAX_MEMOS > 0 for attention over Memos, stored in a StateMatrix.
# Attention over both past observations and Memos would require two separate instances of StateMatrix.
# The WMG experiments did not explore this combination, so only one StateMatrix is used at a time.
assert (WMG_MAX_MEMOS == 0) or (WMG_MAX_OBS == 0)
STATE_VECTORS = WMG_MAX_MEMOS + WMG_MAX_OBS
if WMG_MAX_MEMOS:
    WMG_MEMO_SIZE = cf.val("WMG_MEMO_SIZE")

# The WMG code was refactored in minor ways before the Sokoban experiments. Here's a summary:
#   V1:  Used for Pathfinding and BabyAI.
#        Supports a single factor type.
#        Actor and critic networks are separate.
#        The Memo matrix is initialized with zero vectors.
#        Age vectors are 1-hot.
#   V2:  Used for Sokoban.
#        Supports multiple factor types, one of which is the Core vector.
#        Actor and critic networks share their initial layer.
#        The Memo matrix is initialized empty.
#        Age vectors are 1-hot, concatenated with the normalized age scalar.


class StateMatrix(object):
    ''' Used to store either Factors (from past observations) or Memos. '''
    def __init__(self, max_vectors, vector_len):
        self.vector_len = vector_len
        self.max_vectors = max_vectors
        self.num_vectors = 0  # Capped at self.max_vectors.
        self.vectors = [torch.tensor(np.zeros(self.vector_len, np.float32)) for n in range(self.max_vectors)]
        self.next_index = 0  # Where to store the next vector, overwriting the oldest.

    def get_vector(self, i):
        assert (i >= 0) and (i < self.max_vectors)
        if V2:
            i += 1
        return self.vectors[(self.next_index - i + self.max_vectors) % self.max_vectors]

    def add_vector(self, vector):
        self.vectors[self.next_index] = vector
        self.next_index = (self.next_index + 1) % self.max_vectors
        if self.num_vectors < self.max_vectors:
            self.num_vectors += 1

    def clone(self):
        g = StateMatrix(STATE_VECTORS, self.vector_len)
        for i in range(self.max_vectors):
            g.vectors[i] = self.vectors[i].clone()
        g.next_index = self.next_index
        g.num_vectors = self.num_vectors
        return g

    def detach_from_history(self):
        for i in range(self.max_vectors):
            self.vectors[i] = self.vectors[i].detach()


class WMG_Network(nn.Module):
    ''' Working Memory Graph '''
    def __init__(self, observation_space_size, action_space_size):
        super(WMG_Network, self).__init__()
        self.factored_observations = \
            isinstance(observation_space_size, list) or \
            isinstance(observation_space_size, tuple) or \
            isinstance(observation_space_size, Graph)
        self.tfm_vec_size = TFM_NUM_ATTENTION_HEADS * TFM_ATTENTION_HEAD_SIZE

        if V2:
            assert self.factored_observations
            self.factor_sizes = observation_space_size.entity_type_sizes
            self.embedding_layers = nn.ModuleList()
            for factor_size in self.factor_sizes:
                self.embedding_layers.append(LinearLayer(factor_size, self.tfm_vec_size))
        else:
            if self.factored_observations:
                self.core_vec_size = observation_space_size[0]
                self.factor_vec_size = observation_space_size[1]
            else:
                self.core_vec_size = observation_space_size

        if WMG_MAX_OBS:
            assert not self.factored_observations
            self.state_vector_len = observation_space_size
        else:
            self.state_vector_len = WMG_MEMO_SIZE

        if V2:
            if STATE_VECTORS > 0:
                self.memo_embedding_layer = LinearLayer(WMG_MEMO_SIZE, self.tfm_vec_size)
        else:
            self.age_vec_len = STATE_VECTORS
            self.core_embedding_layer = LinearLayer(self.core_vec_size, self.tfm_vec_size)
            if self.factored_observations:
                self.factor_embedding_layer = LinearLayer(self.factor_vec_size, self.tfm_vec_size)
            self.memo_embedding_layer = LinearLayer(self.state_vector_len + self.age_vec_len, self.tfm_vec_size)

        # Prepare the age vectors.
        self.age = []
        if V2:
            # self.age contains a total of STATE_VECTORS+2 vectors, each of size STATE_VECTORS+2.
            # self.age[0] is for the current time step.
            # self.age[STATE_VECTORS] is for the oldest time step in the history.
            # self.age[STATE_VECTORS+1] is for out-of-range ages.
            # One additional element is copied into any age vector to hold the raw age value itself.
            # Each vector is of size STATE_VECTORS+2: a 1-hot vector of size STATE_VECTORS+1, and the age scalar.
            self.max_age_index = STATE_VECTORS + 1
            age_vector_size = STATE_VECTORS + 2
            for i in range(self.max_age_index):
                pos = np.zeros(age_vector_size, np.float32)
                pos[i] = 1.
                self.age.append(pos)
            self.age.append(np.zeros(age_vector_size, np.float32))
            self.age_embedding_layer = LinearLayer(age_vector_size, self.tfm_vec_size)
        else:
            for i in range(STATE_VECTORS):
                pos = np.zeros(self.age_vec_len, np.float32)
                pos[i] = 1.
                self.age.append(torch.tensor(pos))

        self.tfm = Transformer(TFM_NUM_ATTENTION_HEADS, TFM_ATTENTION_HEAD_SIZE,
                               TFM_NUM_LAYERS, TFM_HIDDEN_SIZE)

        if not V2:
            self.actor_critic_layers = ActorCriticLayers(self.tfm_vec_size, 2, AC_HIDDEN_LAYER_SIZE, action_space_size)

        if WMG_MAX_MEMOS > 0:
            self.memo_creation_layer = LinearLayer(self.tfm_vec_size, WMG_MEMO_SIZE)

        if V2:
            self.actor_critic_layers = SharedActorCriticLayers(self.tfm_vec_size, 2, AC_HIDDEN_LAYER_SIZE, action_space_size)

    def add_age_embedding(self, vector_in, age):
        age_vec = self.age[min(age, self.max_age_index)]
        age_vec[-1] = age / STATE_VECTORS  # Include the normalized age scalar to convey order.
        age_embedding = self.age_embedding_layer(torch.tensor(age_vec))
        return vector_in + age_embedding

    def forward(self, observation, old_matrix):
        # Embed the core vector, and any factor vectors.
        embedded_vec_list = []
        if self.factored_observations:
            if V2:
                for entity in observation.entities:
                    embedded_node_vec = self.embedding_layers[entity.type](torch.tensor(entity.data))
                    embedded_vec_list.append(embedded_node_vec)
            else:
                embedded_vec_list.append(self.core_embedding_layer(torch.tensor(np.float32(observation[0]))))
                for factor_vec in observation[1:]:
                    embedded_vec_list.append(self.factor_embedding_layer(torch.tensor(np.float32(factor_vec))))
        else:
            observations_tens = torch.tensor(np.float32(observation))
            embedded_vec_list.append(self.core_embedding_layer(observations_tens))

        # Embed any state vectors.
        if V2:
            for i in range(old_matrix.num_vectors):
                embedded_vector = self.add_age_embedding(self.memo_embedding_layer(old_matrix.get_vector(i)), i)
                embedded_vec_list.append(embedded_vector)
        else:
            for i in range(STATE_VECTORS):
                embedded_vector = self.memo_embedding_layer(torch.cat((old_matrix.get_vector(i), self.age[i])))
                embedded_vec_list.append(embedded_vector)

        # Reshape the tensor for the Transformer to handle.
        graph_input = torch.cat(embedded_vec_list).view(1, len(embedded_vec_list), self.tfm_vec_size)

        # Run the data through the Transformer.
        graph_output = self.tfm(graph_input)

        # Pass only the core (first) column's output to the subsequent layers.
        tfm_output = graph_output[:,0,:]

        # Update any state.
        if STATE_VECTORS > 0:
            if WMG_MAX_MEMOS:
                # Create a new memo.
                new_vector = torch.tanh(self.memo_creation_layer(tfm_output))[0]
            else:
                # Create a new observation factor.
                new_vector = torch.tanh(observations_tens)
            new_matrix = old_matrix.clone()
            new_matrix.add_vector(new_vector)
        else:
            new_matrix = None

        # Prepare the output.
        value_est, policy = self.actor_critic_layers(tfm_output)
        return value_est, policy, new_matrix

    def init_state(self):
        if STATE_VECTORS > 0:
            return StateMatrix(STATE_VECTORS, self.state_vector_len)
        else:
            return None

    def detach_from_history(self, matrix):
        if matrix is not None:
            matrix.detach_from_history()
        return matrix
