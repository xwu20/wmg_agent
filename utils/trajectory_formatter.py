# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np

from utils.config_handler import cf
TRAJ_HISTORY_LENGTH = cf.val("TRAJ_HISTORY_LENGTH")
TRAJ_INCLUDE_ACTIONS = cf.val("TRAJ_INCLUDE_ACTIONS")
TRAJ_INCLUDE_REWARDS = cf.val("TRAJ_INCLUDE_REWARDS")

CORE_FACTOR_TYPE = 0
OBSERVATION_FACTOR_TYPE = 1
ACTION_FACTOR_TYPE = 2
REWARD_FACTOR_TYPE = 3


class Step(object):
    def __init__(self, observation=None, action=None, reward=None, done=None):
        self.overwrite(observation, action, reward, done)

    def overwrite(self, observation=None, action=None, reward=None, done=None):
        self.observation = observation  # ndarray (float32)
        self.action = action # ndarray (float32)
        self.reward = reward # float
        self.done = done  # int

    def copy(self):
        c = Step(self.observation.copy(), self.action.copy(), self.reward, self.done)
        return c


class History(object):
    def __init__(self):
        self.steps = [Step() for count in range(TRAJ_HISTORY_LENGTH)]
        self.next_index = 0  # Where to store the next step, overwriting the oldest.
        self.num_steps_added = 0

    def add_step(self, observation, action=None, reward=None):
        self.steps[self.next_index].overwrite(observation, action, reward)
        self.next_index = (self.next_index + 1) % TRAJ_HISTORY_LENGTH
        if self.num_steps_added < TRAJ_HISTORY_LENGTH:
            self.num_steps_added += 1

    def get_step(self, i):
        assert (i >= 0) and (i < TRAJ_HISTORY_LENGTH)
        return self.steps[(self.next_index - (i + 1) + TRAJ_HISTORY_LENGTH) % TRAJ_HISTORY_LENGTH]


class TrajectoryFormatter(object):
    def __init__(self, observation_space_size, action_space_size):
        self.factor_sizes = []
        self.observation_vec_size = observation_space_size
        self.factor_sizes.append(self.observation_vec_size)  # Send the current observation to the core.
        self.factor_sizes.append(self.observation_vec_size)  # History observations.
        if TRAJ_INCLUDE_ACTIONS:
            self.action_vec_size = action_space_size  # Just a 1-hot vector for now.
            self.factor_sizes.append(self.action_vec_size)  # History actions.
        if TRAJ_INCLUDE_REWARDS:
            self.factor_sizes.append(1)  # History rewards.
        self.history = None
        self.last_observation = None

    def format_agent_observation(self, action, reward, done, observation):
        if self.last_observation is not None:
            if TRAJ_INCLUDE_ACTIONS:
                if action is not None:
                    action_vec = np.zeros(self.action_vec_size, np.float32)
                    action_vec[action] = 1.
                    action = action_vec
            self.history.add_step(self.last_observation, action, reward)
        observation = np.float32(observation)
        self.last_observation = observation
        composite_observation = []
        composite_observation.append((CORE_FACTOR_TYPE, None, observation))
        composite_observation.append((OBSERVATION_FACTOR_TYPE, 0, observation))
        for i in range(self.history.num_steps_added):
            step = self.history.get_step(i)
            composite_observation.append((OBSERVATION_FACTOR_TYPE, i+1, step.observation))
            if TRAJ_INCLUDE_ACTIONS:
                composite_observation.append((ACTION_FACTOR_TYPE, i+1, step.action))
            if TRAJ_INCLUDE_REWARDS:
                composite_observation.append((REWARD_FACTOR_TYPE, i+1, step.reward))
        return composite_observation

    def format_example(self, example):
        ex = example[0]
        current_step = ex[0]
        observation = current_step.observation.copy()
        composite_observation = []
        composite_observation.append((CORE_FACTOR_TYPE, None, observation))
        composite_observation.append((OBSERVATION_FACTOR_TYPE, 0, observation))
        for i in range(len(ex) - 1):
            step = ex[i+1]
            composite_observation.append((OBSERVATION_FACTOR_TYPE, i+1, step.observation))
            if TRAJ_INCLUDE_ACTIONS:
                composite_observation.append((ACTION_FACTOR_TYPE, i+1, step.action))
            if TRAJ_INCLUDE_REWARDS:
                composite_observation.append((REWARD_FACTOR_TYPE, i+1, step.reward))
        return composite_observation

    def reset_state(self):
        self.history = History()
        self.last_observation = None
