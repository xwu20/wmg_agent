import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
import numpy as np
import copy
from torch.autograd import Variable as var



from utils.spec_reader import spec

hidden_size = spec.val("DNC_HIDDEN_SIZE")
memory_size = spec.val("DNC_MEMORY_SIZE")
word_size = spec.val("DNC_WORD_SIZE")
num_write_heads = spec.val("DNC_NUM_WRITE_HEADS")
num_read_heads = spec.val("DNC_READ_HEADS")
num_read_modes = 1 + 2*num_write_heads
batch_size = 1

TemporalLinkageState = collections.namedtuple('TemporalLinkageState',
											  ('link', 'precedence_weights'))


AccessState = collections.namedtuple('AccessState', (
	'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))

_EPSILON = 0.001


def CosineSimilarity(a, b, dimA=2, dimB=2, normBy=2):
	"""Batchwise Cosine distance
	Cosine distance
	Arguments:
		a {Tensor} -- A 3D Tensor (b * m * w)
		b {Tensor} -- A 3D Tensor (b * r * w)
	Keyword Arguments:
		dimA {number} -- exponent value of the norm for `a` (default: {2})
		dimB {number} -- exponent value of the norm for `b` (default: {1})
	Returns:
		Tensor -- Batchwise cosine distance (b * r * m)
	"""
	#a_norm = torch.norm(a, normBy, dimA, keepdim=True).expand_as(a) + _EPSILON
	a_norm_unexpanded = torch.norm(a, normBy, dimA, keepdim=True) + _EPSILON

	#b_norm = torch.norm(b, normBy, dimB, keepdim=True).expand_as(b) + _EPSILON
	b_norm_unexpanded = torch.norm(b, normBy, dimB, keepdim=True) + _EPSILON
	dot = torch.bmm(a, b.transpose(1, 2)).transpose(1, 2)

	#prod_norm = torch.bmm(a_norm, b_norm.transpose(1, 2)).transpose(1, 2) + _EPSILON
	prod_norm_unexpanded = torch.bmm(b_norm_unexpanded, a_norm_unexpanded.transpose(1, 2)) + _EPSILON
	# print("product norm is")
	# print(prod_norm)

	x = dot / (prod_norm_unexpanded)
	# apply_dict(locals())
	return x


def Softmax_axis(input, axis=1):
	"""Softmax on an axis
	Softmax on an axis
	Arguments:
		input {Tensor} -- input Tensor
	Keyword Arguments:
		axis {number} -- axis on which to take softmax on (default: {1})
	Returns:
		Tensor -- Softmax output Tensor
	"""
	input_size = input.size()

	trans_input = input.transpose(axis, len(input_size) - 1)
	trans_size = trans_input.size()

	input_2d = trans_input.contiguous().view(-1, trans_size[-1])
	soft_max_2d = torch.nn.functional.softmax(input_2d, -1)
	soft_max_nd = soft_max_2d.view(*trans_size)
	return soft_max_nd.transpose(axis, len(input_size) - 1)

def erase_and_write(memory, address, reset_weights, values):
	weighted_resets = torch.unsqueeze(address, 3)*torch.unsqueeze(reset_weights, 2)
	reset_gate = torch.prod(1-weighted_resets, 1)
	memory = memory * reset_gate
	memory = memory + torch.matmul(torch.transpose(address, 1,2), values)

	return memory

class MemoryModule(nn.Module):

	def __init__(self):
		super(MemoryModule, self).__init__()

		self.write_vectors_layer = nn.Linear(hidden_size, num_write_heads*word_size)
		self.erase_vectors_layer = nn.Linear(hidden_size, num_write_heads*word_size)
		self.free_gate_layer = nn.Linear(hidden_size, num_read_heads)
		self.allocation_gate_layer = nn.Linear(hidden_size, num_write_heads)
		self.write_gate_layer = nn.Linear(hidden_size, num_write_heads)
		self.read_mode_layer = nn.Linear(hidden_size, num_read_heads*num_read_modes)
		self.write_keys_layer = nn.Linear(hidden_size, num_write_heads*word_size)
		self.write_strengths_layer = nn.Linear(hidden_size, num_write_heads)
		self.read_keys_layer = nn.Linear(hidden_size, num_read_heads*word_size)
		self.read_strengths_layer = nn.Linear(hidden_size, num_read_heads)

	def forward(self, x, prev_state):

		read_output = self.read_inputs(x)

		usage = self.computeUsage(read_output, prev_state)

		write_weights = self.computeWrite_weights(read_output, prev_state.memory, usage)
		

		memory = erase_and_write(prev_state.memory, write_weights,
			read_output['erase_vector'], read_output['write_vector'])

		linkage_state = self.linkage(write_weights, prev_state.linkage)


		read_words, read_weights = self.computeReadWeights(read_output, memory, prev_state, linkage_state)

		return (read_words, AccessState(memory=memory, read_weights=read_weights,
			write_weights=write_weights, linkage=linkage_state, usage=usage))




	def read_inputs(self, x):
		write_vector = torch.reshape(self.write_vectors_layer(x), (num_write_heads, word_size))
		erase_vector = torch.reshape(torch.sigmoid(self.erase_vectors_layer(x)), (num_write_heads, word_size))

		free_gate = torch.sigmoid(self.free_gate_layer(x))

		allocation_gate = torch.sigmoid(self.allocation_gate_layer(x))

		write_gate = torch.sigmoid(self.write_gate_layer(x))

		read_mode = torch.reshape(torch.sigmoid(self.read_mode_layer(x)), (num_read_heads, num_read_modes))

		write_keys = torch.reshape(torch.sigmoid(self.write_keys_layer(x)), (num_write_heads, word_size))

		write_strengths = self.write_strengths_layer(x)

		read_keys = torch.reshape(self.read_keys_layer(x), (num_read_heads, word_size))

		read_strengths = self.read_strengths_layer(x)


		result = {
		'write_vector': write_vector,
		'erase_vector': erase_vector,
		'free_gate': free_gate,
		'allocation_gate': allocation_gate,
		'write_gate': write_gate,
		'read_mode': read_mode,
		'write_keys': write_keys,
		'write_strengths': write_strengths,
		'read_keys':read_keys,
		'read_strengths':read_strengths
		}

		return result


	def computeUsage(self, inputs, prev_state):


		#write_weights = 1-torch.prod(new_write_weights, 1)
		usage = prev_state.usage + (1-prev_state.usage)*(1-torch.prod(prev_state.write_weights.detach(), 1))

		#should there be 1- in this phi expression
		phi = torch.prod(torch.unsqueeze(inputs['free_gate'], -1) * prev_state.read_weights, 1)

		return usage*phi


	def allocate(self, usage, write_gate):
		# ensure values are not too small prior to cumprod.
		usage = _EPSILON + (1 - _EPSILON) * usage
		batch_size = usage.size(0)
		# free list
		sorted_usage, phi = torch.topk(usage, memory_size, dim=1, largest=False)

		# cumprod with exclusive=True
		# https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
		v = var(sorted_usage.data.new(batch_size, 1).fill_(1))
		cat_sorted_usage = torch.cat((v, sorted_usage), 1)
		prod_sorted_usage = torch.cumprod(cat_sorted_usage, 1)[:, :-1]

		sorted_allocation_weights = (1 - sorted_usage) * prod_sorted_usage.squeeze()

		# construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
		_, phi_rev = torch.topk(phi, k=memory_size, dim=1, largest=False)
		allocation_weights = sorted_allocation_weights.gather(1, phi_rev.long())

		return allocation_weights.unsqueeze(1)

	def write_weighting(self, write_content_weights, allocation_weights, write_gate, allocation_gate):
		ag = allocation_gate.unsqueeze(-1)
		wg = write_gate.unsqueeze(-1)

		return wg * (ag * allocation_weights + (1 - ag) * write_content_weights)


	def computeWrite_weights(self, read_inputs, memory, usage):
		d = CosineSimilarity(memory, torch.unsqueeze(read_inputs['write_keys'], 0))
		write_content_weights = Softmax_axis(d * read_inputs['write_strengths'].unsqueeze(2), 2)
		alloc_weights = self.allocate(usage, read_inputs['allocation_gate'] * read_inputs['write_gate'])
		# print("alloc_weights")
		# print(alloc_weights)
		# print("usaage iis")
		# print(usage)

		write_weights = self.write_weighting(write_content_weights, alloc_weights, read_inputs['write_gate'], read_inputs['allocation_gate'])
		# print("write weights is")
		# print(write_weights)
		return write_weights

	# def _allocation(self, usage_in):

	# 	usage = _EPSILON + (1 - _EPSILON) * usage_in


	# 	sorted_usage, indices = torch.topk(usage, memory_size, largest=False, sorted=True)

	# 	prod_sorted_usage = torch.cumprod(sorted_usage, axis=1)
	# 	exclusive_prod_sorted_usage = torch.ones(1,memory_size)
	# 	exclusive_prod_sorted_usage[0][1:] = prod_sorted_usage[0][:-1]

	# 	sorted_allocation = (1-sorted_usage)*exclusive_prod_sorted_usage

	# 	final_allocation = torch.zeros((1, memory_size))
	# 	#undo the initial permutation
	# 	for i in range(memory_size):
	# 		a_index = indices[0][i]
	# 		final_allocation[0][a_index] = sorted_allocation[0][i]

	# 	return final_allocation

	def linkage(self, write_weights, prev_linkage):

		prev_precedence_weights_tensor = prev_linkage.precedence_weights
		prev_link = prev_linkage.link
		batch_size = prev_linkage.link.shape[0]
		write_weights_i = torch.unsqueeze(write_weights, 3)
		write_weights_j = torch.unsqueeze(write_weights, 2)
		prev_precedence_weights_j = torch.unsqueeze(prev_precedence_weights_tensor, 2)
		prev_link_scale = 1 - write_weights_i - write_weights_j
		new_link = write_weights_i * prev_precedence_weights_j
		link = prev_link_scale * prev_link + new_link

		for i in range(num_write_heads):
			link[0][i].fill_diagonal_(0)

		write_sum_raw = torch.sum(write_weights, 2)
		write_sum = torch.unsqueeze(write_sum_raw, 2)

		precedence_weights = (1 - write_sum) * prev_precedence_weights_tensor + write_weights

		return TemporalLinkageState(link=link, precedence_weights=precedence_weights)

	def computeReadWeights(self, inputs, memory, prev_state, linkage_state):
		d = CosineSimilarity(memory, torch.unsqueeze(inputs['read_keys'], 0))

		content_weights = Softmax_axis(d * inputs['read_strengths'].unsqueeze(2), 2)

		link = linkage_state.link


		expanded_read_weights = torch.stack((prev_state.read_weights, prev_state.read_weights), dim=1)

		result = torch.matmul(expanded_read_weights, torch.transpose(link, 2, 3))
		result2 = torch.matmul(expanded_read_weights, link)

		forward_weights = result.permute(0, 2, 1, 3)
		backward_weights = result2.permute(0, 2, 1, 3)

		batched_read_mode = torch.unsqueeze(inputs['read_mode'], 0)

		backward_mode = batched_read_mode[:,:, :num_write_heads]
		forward_mode = (batched_read_mode[:,:, num_write_heads:2 * num_write_heads])
		content_mode = batched_read_mode[:,:, 2 * num_write_heads]

		read_weights = (
		  torch.unsqueeze(content_mode, 2) * content_weights + torch.sum(
			  torch.unsqueeze(forward_mode, 3) * forward_weights, 2) +
		  torch.sum(torch.unsqueeze(backward_mode, 3) * backward_weights, 2))

		read_words = torch.matmul(read_weights, memory)
		return read_words, read_weights

	def state_size(self):
		"""Returns a tuple of the shape of the state tensors."""
		return AccessState(
			memory=[memory_size, word_size],
			read_weights=[num_reads, memory_size],
			write_weights=[num_writes, memory_size],
			linkage=TemporalLinkageState(
				link=[num_writes, memory_size, memory_size],
				precedence_weights=[num_writes, memory_size],),
			usage=[memory_size])

		

