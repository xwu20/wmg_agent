# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config_handler import cf
REFACTORED = cf.val("REFACTORED")
AGENT_RANDOM_SEED = cf.val("AGENT_RANDOM_SEED")
BPTT_PERIOD = cf.val("BPTT_PERIOD")
LEARNING_RATE = cf.val("LEARNING_RATE")
DISCOUNT_FACTOR = cf.val("DISCOUNT_FACTOR")
GRADIENT_CLIP = cf.val("GRADIENT_CLIP")
WEIGHT_DECAY = cf.val("WEIGHT_DECAY")
AGENT_NET = cf.val("AGENT_NET")
ENTROPY_REG = cf.val("ENTROPY_REG")
REWARD_SCALE = cf.val("REWARD_SCALE")
ADAM_EPS = cf.val("ADAM_EPS")
from agents.networks.shared.general import LinearActorCriticLayer
from utils.graph import Graph
ANNEAL_LR = cf.val("ANNEAL_LR")
if ANNEAL_LR:
    LR_GAMMA = cf.val("LR_GAMMA")
    from torch.optim.lr_scheduler import StepLR

torch.manual_seed(AGENT_RANDOM_SEED)


class A3cAgent(object):
    def __init__(self, observation_space_size, action_space_size):
        self.factored_observations = isinstance(observation_space_size, list) or isinstance(observation_space_size, tuple) or isinstance(observation_space_size, Graph)
        self.internal_observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        if REFACTORED:
            self.network = self.create_network()
        else:
            self.network = self.create_network_old()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
        if ANNEAL_LR:
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=LR_GAMMA)
        print("{:11,d} trainable parameters".format(self.count_parameters(self.network)))

    def create_network(self):
        # if AGENT_NET == "GRU_Network":
        #     from agents.networks.gru import GRU_Network
        #     core = GRU_Network(self.internal_observation_space_size, self.action_space_size)
        if AGENT_NET == "WMG_Network_S":
            from agents.networks.wmg_s import WMG_Network_S
            core = WMG_Network_S(self.internal_observation_space_size, self.action_space_size, self.factored_observations)
        else:
            assert False  # The specified agent network was not found.
        actor_critic_layers = LinearActorCriticLayer(core.output_size, self.action_space_size)
        return AgentModel(core, actor_critic_layers)

    def create_network_old(self):
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
        self.net_state = self.network.init_state()
        self.last_action = None
        self.reset_adaptation_state()
        return 0

    def load_model(self, input_model):
        state_dict = torch.load(input_model)
        self.network.load_state_dict(state_dict)
        print('loaded agent model from {}'.format(input_model))

    def save_model(self, output_model):
        torch.save(self.network.state_dict(), output_model)

    def step(self, observation):
        self.last_observation = observation
        self.value_tensor, logits, self.net_state = self.network(self.last_observation, self.net_state)
        self.logp_tensor = F.log_softmax(logits, dim=-1)
        action_probs = torch.exp(self.logp_tensor)
        self.action_tensor = action_probs.multinomial(num_samples=1).data[0]
        self.last_action = self.action_tensor.numpy()[0]
        return self.last_action

    def adapt(self, reward, done, next_observation):
        reward *= REWARD_SCALE

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
        # Reset_state will get called after this.
        self.train(0.)

    def adapt_on_end_of_sequence(self, next_observation):
        # Peek at the state value of the next observation, for TD calculation.
        next_value, _, _ = self.network(next_observation, self.net_state)

        # Train.
        self.train(next_value.squeeze().data.numpy())

        # Stop the gradients from flowing back into this window that we just trained on.
        self.net_state = self.network.detach_from_history(self.net_state)

        self.reset_adaptation_state()

    def train(self, next_value):
        loss = self.loss_function(next_value, torch.cat(self.values), torch.cat(self.logps), torch.cat(self.actions), np.asarray(self.rewards))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), GRADIENT_CLIP)
        self.optimizer.step()

    def anneal_lr(self):
        print('Scaling down the learning rate by {}'.format(LR_GAMMA))
        self.scheduler.step()

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

class AgentModel(nn.Module):
    def __init__(self, core, actor_critic):
        super(AgentModel, self).__init__()
        self.repres = core  # Network core representation module.
        self.actor_critic = actor_critic

    def forward(self, input, old_state):
        h, new_state = self.repres(input, old_state)  # The base network may be recurrent.
        value_est, policy = self.actor_critic(h)
        return value_est, policy, new_state

    def init_state(self):
        return self.repres.init_state()

    def detach_from_history(self, old_state):
        new_state = self.repres.detach_from_history(old_state)
        return new_state
