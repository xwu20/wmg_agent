# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.config_handler import cf
BPTT_PERIOD = cf.val("BPTT_PERIOD")
LEARNING_RATE = cf.val("LEARNING_RATE")
DISCOUNT_FACTOR = cf.val("DISCOUNT_FACTOR")
GRADIENT_CLIP = cf.val("GRADIENT_CLIP")
WEIGHT_DECAY = cf.val("WEIGHT_DECAY")
OBS_FEEDBACK = cf.val("OBS_FEEDBACK")
AGENT_NET = cf.val("AGENT_NET")
ENTROPY_REG = cf.val("ENTROPY_REG")
REWARD_SCALE = cf.val("REWARD_SCALE")
ADAM_EPS = cf.val("ADAM_EPS")


class A3cAgent(object):
    def __init__(self, agent_name, observation_space_size, action_space_size):
        self.name = agent_name
        self.factored_observations = isinstance(observation_space_size, tuple)
        self.external_observation_space_size = observation_space_size
        self.internal_observation_space_size = observation_space_size
        if OBS_FEEDBACK and not self.factored_observations:
            self.internal_observation_space_size += action_space_size + 1
        self.action_space_size = action_space_size
        self.global_net = None
        if agent_name == '0':
            self.global_net = self.create_network().share_memory()
        self.local_net = self.create_network()
        if self.global_net is not None:
            self.optimizer = torch.optim.Adam(self.global_net.parameters(), lr=LEARNING_RATE,
                                              weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
        print("{:11,d} trainable parameters".format(self.count_parameters(self.local_net)))

    def create_network(self):
        if AGENT_NET == "GRU_Network":
            from agents.networks.gru import GRU_Network
            return GRU_Network(self.internal_observation_space_size, self.action_space_size)
        if AGENT_NET == "WMG_Network":
            from agents.networks.wmg import WMG_Network
            return WMG_Network(self.internal_observation_space_size, self.action_space_size, self.factored_observations)
        assert False  # The specified agent network was not found.

    def count_parameters(self, network):
        return sum(p.numel() for p in network.parameters() if p.requires_grad)

    def reset_adaptation_state(self):
        self.num_training_frames_in_buffer = 0
        self.values = []
        self.logps = []
        self.actions = []
        self.rewards = []

    def reset_state(self):
        self.copy_global_weights()
        self.net_state = self.local_net.init_state()
        self.last_action = None
        self.last_reward = 0.
        self.reset_adaptation_state()
        return 0

    def copy_global_weights(self):
        self.local_net.load_state_dict(self.global_net.state_dict())

    def save_model(self, filename):
        torch.save(self.global_net.state_dict(), filename)

    def load_model(self, filename):
        state_dict = torch.load(filename)
        self.global_net.load_state_dict(state_dict)
        print('loaded')

    def expand_observation(self, external_observation):
        if self.factored_observations or not OBS_FEEDBACK:
            return external_observation
        else:
            id = 0
            expanded_observation = np.zeros(self.internal_observation_space_size)
            expanded_observation[0:self.external_observation_space_size] = external_observation[0:self.external_observation_space_size]
            id += self.external_observation_space_size
            if self.last_action is not None:
                expanded_observation[id + self.last_action] = 1.
            id += self.action_space_size
            expanded_observation[id] = self.last_reward
            id += 1
            assert id == self.internal_observation_space_size
            return expanded_observation

    def step(self, observation):
        self.last_observation = self.expand_observation(observation)
        self.value_tensor, logits, self.net_state = self.local_net(self.last_observation, self.net_state)
        self.logp_tensor = F.log_softmax(logits, dim=-1)
        action_probs = torch.exp(self.logp_tensor)
        self.action_tensor = action_probs.multinomial(num_samples=1).data[0]
        self.last_action = self.action_tensor.numpy()[0]
        return self.last_action

    def adapt(self, reward, done, next_observation):
        reward *= REWARD_SCALE

        # Copy these to be included in the next observation (if the episode doesn't get reset).
        self.last_reward = reward

        # print("OBS = {}".format(self.last_observation))
        # print("ACT = {}".format(self.last_action))
        # print("REW = {}".format(self.last_reward))
        # print()

        # print("g = {}".format(self.global_net.actor_linear.weight.grad))
        # print("w =                                    {}".format(self.global_net.actor_linear.weight))

        # Buffer one frame of data for eventual training.
        self.values.append(self.value_tensor)
        self.logps.append(self.logp_tensor)
        self.actions.append(self.action_tensor)
        self.rewards.append(reward)
        self.num_training_frames_in_buffer += 1

        if done:
            self.adapt_on_end_of_episode()
        elif self.num_training_frames_in_buffer == BPTT_PERIOD:
            self.adapt_on_end_of_sequence(next_observation)

    def adapt_on_end_of_episode(self):
        # Train with a next state value of zero, because there aren't any rewards after the end of the episode.
        # Train. (reset_state will get called after this)
        self.train(0.)

    def adapt_on_end_of_sequence(self, next_observation):
        # Peek at the state value of the next observation, for TD calculation.
        next_expanded_observation = self.expand_observation(next_observation)
        next_value, _, _ = self.local_net(next_expanded_observation, self.net_state)

        # Train.
        self.train(next_value.squeeze().data.numpy())

        # Stop the gradients from flowing back into this window that we just trained on.
        self.net_state = self.local_net.detach_from_history(self.net_state)

        self.reset_adaptation_state()

    def train(self, next_value):
        loss = self.loss_function(next_value, torch.cat(self.values), torch.cat(self.logps), torch.cat(self.actions), np.asarray(self.rewards))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), GRADIENT_CLIP)

        for param, shared_param in zip(self.local_net.parameters(), self.global_net.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad  # sync gradients with global network
        self.optimizer.step()

        self.copy_global_weights()

    def loss_function(self, next_value, values, logps, actions, rewards):
        td_target = next_value
        np_values = values.view(-1).data.numpy()
        buffer_size = len(rewards)
        td_targets = np.zeros(buffer_size)
        advantages = np.zeros(buffer_size)

        # Populate the td_target array (for value update) and the advantage array (for policy update).
        # Note that value errors should be backpropagated through the current value calculation for value updates,
        # but not for policy updates.
        for i in range(buffer_size - 1, -1, -1):
            td_target = rewards[i] + DISCOUNT_FACTOR * td_target
            advantage = td_target - np_values[i]
            td_targets[i] = td_target
            advantages[i] = advantage

        chosen_action_log_probs = logps.gather(1, actions.view(-1, 1))
        advantages_tensor = torch.FloatTensor(advantages.copy())
        policy_losses = chosen_action_log_probs.view(-1) * advantages_tensor
        policy_loss = -policy_losses.sum()

        td_targets_tensor = torch.FloatTensor(td_targets.copy())

        value_losses = td_targets_tensor - values[:, 0]
        value_loss = value_losses.pow(2).sum()

        entropy_losses = -logps * torch.exp(logps)
        entropy_loss = entropy_losses.sum()
        return policy_loss + 0.5 * value_loss - ENTROPY_REG * entropy_loss

