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


		new_prev_state = DNCState(access_output[0], access_output[1], (hn, cn))
		return policy, value_est, new_prev_state

	def init_state(self):
		#to set up the initial access state
		#initial_memory = torch.from_numpy(np.random.rand(batch_size, memory_size, word_size)).float()

		initial_memory = torch.zeros(batch_size, memory_size, word_size)
		torch.nn.init.uniform_(initial_memory, a=0.0, b=1.0)
		initial_read_weights = torch.empty(batch_size, num_read_heads, memory_size)
		torch.nn.init.uniform_(initial_read_weights, a=0.0, b=1/float(memory_size))
		#initial_read_weights = torch.nn.init.constant_(initial_read_weights, 1/float(memory_size))
		#initial_read_weights = torch.from_numpy(np.random.rand(batch_size, num_read_heads, memory_size)).float()
		initial_write_weights = torch.empty(batch_size, num_write_heads, memory_size)
		torch.nn.init.uniform_(initial_write_weights, a=0.0, b=1/float(memory_size))
		#initial_write_weights = torch.nn.init.constant_(initial_write_weights, 1/float(memory_size))
		#initial_write_weights = torch.from_numpy(np.random.rand(batch_size, num_write_heads, memory_size)).float()
		#initial_linkage = np.random.rand(batch_size, num_write_heads, memory_size, memory_size)
		initial_usage = torch.empty(batch_size, memory_size)
		torch.nn.init.uniform_(initial_usage, a=0.0, b=1.0)

		#initial_usage = torch.nn.init.constant_(initial_usage, 1/float(memory_size))
		
		initial_linkage_link = torch.zeros(batch_size, num_write_heads, memory_size, memory_size)
		#initial_linkage_link = torch.from_numpy(np.random.rand(batch_size, num_write_heads, memory_size, memory_size)).float()
		initial_linkage_precendence_weights = torch.zeros(batch_size, num_write_heads, memory_size)
		#initial_linkage_precendence_weights = torch.from_numpy(np.random.rand(batch_size, num_write_heads, memory_size)).float()

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


	# def forward(self, x, prev_state):
	# 	prev_access_output = prev_state.access_output
	# 	prev_access_state = prev_state.access_state
	# 	prev_controller_state = prev_state.controller_state


	# 	batch_flatten = snt.BatchFlatten()
	# 	controller_input = tf.concat([batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

	# 	controller_output, controller_state = self.controller(controller_input, prev_controller_state)
	# 	access_output, access_state = self._access(controller_output,
 #                                               prev_access_state)
	# 	output = tf.concat([controller_output, batch_flatten(access_output)], 1)

	# 	output = nn.Linear(output_size=self._output_size.as_list()[0])

	# 	policy, value_est  = self.actor_critic_layers(output)

	# 	return policy, value_est, DNCState(
	# 		access_output=access_output,
	# 		access_state=access_state,
	# 		controller_state=controller_state)



    #controller_output = self._clip_if_enabled(controller_output)
    #controller_state = tf.contrib.framework.nest.map_structure(self._clip_if_enabled, controller_state)

    
    #output = self._clip_if_enabled(output)

    