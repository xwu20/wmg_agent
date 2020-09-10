import torch
import torch.nn as nn
import numpy as np

from agents.networks.shared.general import LinearLayer
from agents.networks.shared.general import SeparateActorCriticLayers
from agents.networks.shared.general import SharedActorCriticLayers
from agents.networks.shared.memory import MemoryModule
from utils.graph import Graph

from utils.spec_reader import spec
import collections

hidden_size = spec.val("DNC_HIDDEN_SIZE")

TemporalLinkageState = collections.namedtuple('TemporalLinkageState',
											  ('link', 'precedence_weights'))



DNCState = collections.namedtuple('DNCState', ('access_output', 'access_state',
                                               'controller_state'))

AccessState = collections.namedtuple('AccessState', (
	'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))

AC_HIDDEN_LAYER_SIZE = spec.val("AC_HIDDEN_LAYER_SIZE")

hidden_size = spec.val("DNC_HIDDEN_SIZE")
memory_size = spec.val("DNC_MEMORY_SIZE")
word_size = spec.val("DNC_WORD_SIZE")
num_write_heads = spec.val("DNC_NUM_WRITE_HEADS")
num_read_heads = spec.val("DNC_READ_HEADS")
num_read_modes = 1 + 2*num_write_heads
new_size = num_read_heads*word_size

batch_size = 1

class DNC_Network(nn.Module):

	def __init__(self, observation_space, action_space_size):
		super(DNC_Network, self).__init__()


		self.controller = nn.LSTM(observation_space, hidden_size)
		self.access = MemoryModule()
		

		self.actor_critic_layers = SeparateActorCriticLayers(new_size, 2, AC_HIDDEN_LAYER_SIZE, action_space_size)

	def forward(self, obs, prev_state):
		print("forward again")
		prev_controller_state = prev_state.controller_state
		prev_access_state = prev_state.access_state

		obs_tens = torch.tensor(np.float32(obs))
		obs_tens = torch.flatten(obs_tens)
		tens_tens = torch.FloatTensor(obs_tens).unsqueeze(0)
		tens = torch.FloatTensor(tens_tens).unsqueeze(0)
		controller_output, (hn, cn) = self.controller(tens, prev_controller_state)
		controller_output = torch.squeeze(controller_output, 0)

		access_output = self.access(controller_output, prev_access_state)
		memory_output = access_output[0].view(1, new_size)

		policy, value_est  = self.actor_critic_layers(memory_output)

		access_output_0_clone = access_output[0].clone()
		access_output_1_memory_clone = access_output[1].memory.clone()
		access_output_1_read_weights_clone = access_output[1].read_weights.clone()
		access_output_1_write_weights_clone = access_output[1].write_weights.clone()
		access_output_1_linkage_link_clone = access_output[1].linkage.link.clone()
		access_output_1_linkage_precendence_weights_clone = access_output[1].linkage.precedence_weights.clone()
		access_output_1_usage_clone = access_output[1].usage.clone()

		linkage_clone = TemporalLinkageState(access_output_1_linkage_link_clone, access_output_1_linkage_precendence_weights_clone)

		access_state_clone = AccessState(access_output_1_memory_clone, access_output_1_read_weights_clone, access_output_1_write_weights_clone, linkage_clone, access_output_1_usage_clone)

		(hn_clone, cn_clone) = (hn.clone(), cn.clone())

		#new_prev_state = DNCState(access_output[0], access_output[1], (hn, cn))
		new_prev_state = DNCState(access_output_0_clone, access_state_clone, (hn_clone, cn_clone))
		return policy, value_est, new_prev_state

	def init_state(self):

		initial_memory = torch.zeros(batch_size, memory_size, word_size)
		initial_read_weights = torch.empty(batch_size, num_read_heads, memory_size)
		torch.nn.init.uniform_(initial_read_weights, a=0.0, b=1/float(memory_size))
		initial_write_weights = torch.empty(batch_size, num_write_heads, memory_size)
		torch.nn.init.uniform_(initial_write_weights, a=0.0, b=1/float(memory_size))
		initial_usage = torch.empty(batch_size, memory_size)
		torch.nn.init.uniform_(initial_usage, a=0.0, b=1.0)
		
		initial_linkage_link = torch.zeros(batch_size, num_write_heads, memory_size, memory_size)
		initial_linkage_precendence_weights = torch.zeros(batch_size, num_write_heads, memory_size)

		initialLinkage = TemporalLinkageState(initial_linkage_link, initial_linkage_precendence_weights)

		initial_access_state = AccessState(initial_memory, initial_read_weights, 
			initial_write_weights, initialLinkage, initial_usage)

		#
		initial_state = DNCState(None, initial_access_state, (torch.zeros(1,1,hidden_size), torch.zeros(1,1,hidden_size)))
		return initial_state
	def detach_from_history(self, state):

		new_access_output = state.access_output.detach()
		new_linkage = TemporalLinkageState(state.access_state.linkage.link.detach(), state.access_state.linkage.precedence_weights.detach())
		new_access_state = AccessState(state.access_state.memory.detach(), state.access_state.read_weights.detach(), state.access_state.write_weights.detach(), new_linkage, state.access_state.usage.detach())

		net_state = []
		for k in state.controller_state:
			net_state.append(k.detach())
		new_detach_state = DNCState(new_access_output, new_access_state, (net_state[0], net_state[1]))
		return new_detach_state

    