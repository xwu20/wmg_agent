# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = test_episodes  # train, test, test_episodes, render
NUM_EPISODES_TO_TEST = 100  # 1000 100
MIN_FINAL_REWARD_FOR_SUCCESS = 1.0
LOAD_MODEL_FROM = models/gru_flat_babyai.pth
SAVE_MODELS_TO = None

# worker.py
ENV = BabyAI_Env
ENV_RANDOM_SEED = 1
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 1
TOTAL_STEPS = 1
ANNEAL_LR = False

# A3cAgent
AGENT_NET = GRU_Network

# BabyAI_Env
BABYAI_ENV_LEVEL = BabyAI-GoToLocal-v0
USE_SUCCESS_RATE = True
SUCCESS_RATE_THRESHOLD = 0.99
HELDOUT_TESTING = False
NUM_TEST_EPISODES = 10000
OBS_ENCODER = Flat
BINARY_REWARD = True

###  HYPERPARAMETERS  (tunable)  ###

# Agents in general
A3C_T_MAX = 4
LEARNING_RATE = 4e-05
DISCOUNT_FACTOR = 0.9
GRADIENT_CLIP = 512.0
ENTROPY_TERM_STRENGTH = 0.02
ADAM_EPS = 1e-12

# A3cAgent
REWARD_SCALE = 2.0
WEIGHT_DECAY = 0.

# RNNs
NUM_RNN_UNITS = 96
OBS_EMBED_SIZE = 512
AC_HIDDEN_LAYER_SIZE = 4096
