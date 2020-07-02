# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_gru_factored_babyai.pth

# worker.py
ENV = BabyAI_Env
ENV_MAJOR_RANDOM_SEED = 1  # Use randint for non-deterministic behavior.
ENV_MINOR_RANDOM_SEED = 0
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 50  # 10000 50
TOTAL_STEPS = 250  # 2500000 250
ANNEAL_LR = False
LR_GAMMA = 0.98

# A3cAgent
REFACTORED = False
AGENT_NET = GRU_Network

# BabyAI_Env
BABYAI_ENV_LEVEL = BabyAI-GoToLocal-v0
USE_SUCCESS_RATE = True
SUCCESS_RATE_THRESHOLD = 0.99
HELDOUT_TESTING = True
NUM_TEST_EPISODES = 10000
OBS_ENCODER = FactoredThenFlattened
BINARY_REWARD = True

###  HYPERPARAMETERS  (tunable)  ###

# GRU_Network ##

# Agents in general
AC_HIDDEN_LAYER_SIZE = 1024
BPTT_PERIOD = 3
LEARNING_RATE = 4e-05
DISCOUNT_FACTOR = 0.95
GRADIENT_CLIP = 256.0
ENTROPY_REG = 0.1
ADAM_EPS = 1e-06
OBS_FEEDBACK = False

# A3cAgent
REWARD_SCALE = 8.0
WEIGHT_DECAY = 0.
APPLY_RELU_TO_OBS_EMBED = False

# RNNs
NUM_RNN_UNITS = 128
OBS_EMBED_SIZE = 1024
