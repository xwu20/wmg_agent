import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class LinearLayer(nn.Module):  # Adds weight initialization options on top of nn.Linear.
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.layer.bias.data.fill_(0.)

    def forward(self, input):
        output = self.layer(input)
        return output


class ResidualLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualLayer, self).__init__()
        self.linear_layer = LinearLayer(input_size, output_size)

    def forward(self, input, prev_input):
        output = self.linear_layer(input)
        output += prev_input
        return output


class LinearActorCriticLayer(nn.Module):
    def __init__(self, input_size, action_space_size):
        super(LinearActorCriticLayer, self).__init__()
        self.critic_linear = LinearLayer(input_size, 1)
        self.actor_linear = LinearLayer(input_size, action_space_size)
        self.actor_linear.layer.weight.data.fill_(0.)

    def forward(self, input):
        value = self.critic_linear(input)
        policy = self.actor_linear(input)
        return value, policy


class ActorCriticLayers(nn.Module):
    def __init__(self, input_size, num_layers, hidden_layer_size, action_space_size):
        super(ActorCriticLayers, self).__init__()
        assert num_layers == 2
        self.critic_linear_1 = LinearLayer(input_size, hidden_layer_size)
        self.critic_linear_2 = LinearLayer(hidden_layer_size, 1)
        self.actor_linear_1 = LinearLayer(input_size, hidden_layer_size)
        self.actor_linear_2 = LinearLayer(hidden_layer_size, action_space_size)
        self.actor_linear_2.layer.weight.data.fill_(0.)

    def forward(self, input):
        value = self.critic_linear_1(input)
        value = F.relu(value)
        value = self.critic_linear_2(value)
        policy = self.actor_linear_1(input)
        policy = F.relu(policy)
        policy = self.actor_linear_2(policy)
        return value, policy


class SharedActorCriticLayers(nn.Module):
    def __init__(self, input_size, num_layers, hidden_layer_size, action_space_size):
        super(SharedActorCriticLayers, self).__init__()
        assert num_layers == 2
        self.linear_1 = LinearLayer(input_size, hidden_layer_size)
        self.critic_linear_2 = LinearLayer(hidden_layer_size, 1)
        self.actor_linear_2 = LinearLayer(hidden_layer_size, action_space_size)
        self.actor_linear_2.layer.weight.data.fill_(0.)

    def forward(self, input):
        shared = self.linear_1(input)
        shared = F.relu(shared)
        value = self.critic_linear_2(shared)
        policy = self.actor_linear_2(shared)
        return value, policy
