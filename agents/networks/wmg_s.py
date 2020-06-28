# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.networks.shared.transformer import Transformer
from agents.networks.shared.general import LinearLayer

# Look up settings from config.py that will be used below.
from utils.config_handler import cf
TFM_ATTENTION_HEAD_SIZE = cf.val("TFM_ATTENTION_HEAD_SIZE")
TFM_NUM_ATTENTION_HEADS = cf.val("TFM_NUM_ATTENTION_HEADS")
TFM_NUM_LAYERS = cf.val("TFM_NUM_LAYERS")
TFM_HIDDEN_SIZE = cf.val("TFM_HIDDEN_SIZE")
WMG2_MAX_AGE = cf.val("WMG2_MAX_AGE")
WMG2_MAX_CONCEPTS = cf.val("WMG2_MAX_CONCEPTS")
if WMG2_MAX_CONCEPTS > 0:  # WMG2_CONCEPT_SIZE will be used only if WMG2_MAX_CONCEPTS > 0.
    WMG2_CONCEPT_SIZE = cf.val("WMG2_CONCEPT_SIZE")
WMG2_OUTPUT_LAYER_SIZE = cf.val("WMG2_OUTPUT_LAYER_SIZE")


class ConceptMatrix(object):
    def __init__(self):
        self.concepts = [torch.tensor(np.zeros(WMG2_CONCEPT_SIZE, np.float32)) for n in range(WMG2_MAX_CONCEPTS)]
        self.next_index = 0  # Where to store the next concept, overwriting the oldest.
        self.num_concepts_added = 0  # Capped at WMG2_MAX_CONCEPTS.

    def get_concept(self, i):
        assert (i >= 0) and (i < WMG2_MAX_CONCEPTS)
        return self.concepts[(self.next_index - (i + 1) + WMG2_MAX_CONCEPTS) % WMG2_MAX_CONCEPTS]

    def add_concept(self, concept):
        self.concepts[self.next_index] = concept
        self.next_index = (self.next_index + 1) % WMG2_MAX_CONCEPTS
        if self.num_concepts_added < WMG2_MAX_CONCEPTS:
            self.num_concepts_added += 1

    def clone(self):
        g = ConceptMatrix()
        for i in range(WMG2_MAX_CONCEPTS):
            g.concepts[i] = self.concepts[i].clone()
        g.next_index = self.next_index
        g.num_concepts_added = self.num_concepts_added
        return g

    def detach_from_history(self):
        for i in range(WMG2_MAX_CONCEPTS):
            self.concepts[i] = self.concepts[i].detach()


class WMG_Network_S(nn.Module):
    def __init__(self, observation_space_size, action_space_size, factored_observations):
        super(WMG_Network_S, self).__init__()
        self.factored_observations = factored_observations
        self.node_vec_size = TFM_NUM_ATTENTION_HEADS * TFM_ATTENTION_HEAD_SIZE
        if WMG2_OUTPUT_LAYER_SIZE > 0:
            self.output_size = WMG2_OUTPUT_LAYER_SIZE
        else:
            self.output_size = self.node_vec_size
        if self.factored_observations:
            self.factor_sizes = observation_space_size.entity_type_sizes
            self.embedding_layers = nn.ModuleList()
            for factor_size in self.factor_sizes:
                self.embedding_layers.append(LinearLayer(factor_size, self.node_vec_size))
        else:
            self.core_embed_layer = LinearLayer(observation_space_size, self.node_vec_size)

        if WMG2_MAX_CONCEPTS > 0:
            self.concept_embed_layer = LinearLayer(WMG2_CONCEPT_SIZE, self.node_vec_size)

        # Prepare the age vectors.
        # self.age contains a total of WMG2_MAX_AGE+2 vectors, each of size WMG2_MAX_AGE+2.
        # self.age[0] is for the current time step.
        # self.age[WMG2_MAX_AGE] is for the oldest time step in the history.
        # self.age[WMG2_MAX_AGE+1] is for out-of-range ages.
        # One additional element is copied into any age vector to hold the raw age value itself.
        # Each vector is of size WMG2_MAX_AGE+2, for a 1-hot vector of size WMG2_MAX_AGE+1, and the raw age itself.
        self.age = []
        self.max_age_index = WMG2_MAX_AGE + 1
        age_vector_size = WMG2_MAX_AGE + 2
        for i in range(self.max_age_index):
            pos = np.zeros(age_vector_size, np.float32)
            pos[i] = 1.
            self.age.append(pos)
        self.age.append(np.zeros(age_vector_size, np.float32))
        self.age_embed_layer = LinearLayer(age_vector_size, self.node_vec_size)

        self.tfm = Transformer(TFM_NUM_ATTENTION_HEADS, TFM_ATTENTION_HEAD_SIZE,
                               TFM_NUM_LAYERS, TFM_HIDDEN_SIZE)

        if WMG2_MAX_CONCEPTS > 0:
            self.concept_creation_layer = LinearLayer(self.node_vec_size, WMG2_CONCEPT_SIZE)

        if WMG2_OUTPUT_LAYER_SIZE > 0:
            self.output_layer = LinearLayer(self.node_vec_size, WMG2_OUTPUT_LAYER_SIZE)

    def add_age_embedding(self, vector_in, age):
        age_vec = self.age[min(age, self.max_age_index)]
        age_vec[-1] = age / WMG2_MAX_AGE  # Include the raw age to convey order.
        age_embedding = self.age_embed_layer(torch.tensor(age_vec))
        return vector_in + age_embedding

    def forward(self, observation, old_concept_matrix):
        # Embed the core vector, and any factor vectors.
        node_vec_list = []
        if self.factored_observations:
            for entity in observation.entities:
                embedded_node_vec = self.embedding_layers[entity.type](torch.tensor(entity.data))
                if entity.age is not None:
                    embedded_node_vec = self.add_age_embedding(embedded_node_vec, factor_age)
                node_vec_list.append(embedded_node_vec)
        else:
            node_vec_list.append(self.core_embed_layer(torch.tensor(np.float32(observation))))

        # Embed any concept vectors.
        if WMG2_MAX_CONCEPTS > 0:
            for i in range(old_concept_matrix.num_concepts_added):
                embedded_concept_vec = self.concept_embed_layer(old_concept_matrix.get_concept(i))
                embedded_concept_vec = self.add_age_embedding(embedded_concept_vec, i)
                node_vec_list.append(embedded_concept_vec)

        # Reshape the tensor for the Transformer to handle.
        graph_input = torch.cat(node_vec_list).view(1, len(node_vec_list), self.node_vec_size)

        # Run the data through the Transformer.
        graph_output = self.tfm(graph_input)

        # Pass only the core (first) node's output to the subsequent layers.
        tfm_output = graph_output[:,0,:]  # From the core node only.

        if WMG2_MAX_CONCEPTS > 0:
            # Create a new concept.
            concept = torch.tanh(self.concept_creation_layer(tfm_output))[0]
            new_concept_matrix = old_concept_matrix.clone()
            new_concept_matrix.add_concept(concept)
        else:
            new_concept_matrix = None

        if WMG2_OUTPUT_LAYER_SIZE > 0:
            output = F.relu(self.output_layer(tfm_output))
        else:
            output = tfm_output

        return output, new_concept_matrix

    def init_state(self):
        if WMG2_MAX_CONCEPTS > 0:
            return ConceptMatrix()
        else:
            return None

    def detach_from_history(self, concept_matrix):
        if concept_matrix is not None:
            concept_matrix.detach_from_history()
        return concept_matrix
