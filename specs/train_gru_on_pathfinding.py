# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_gru_pathfinding.pth

# worker.py
ENV = Pathfinding_Env
ENV_MAJOR_RANDOM_SEED = 1  # Use randint for non-deterministic behavior.
ENV_MINOR_RANDOM_SEED = 0
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 120  # 100000 120
TOTAL_STEPS = 600  # 20000000 600
ANNEAL_LR = False
LR_GAMMA = 0.98

# A3cAgent
REFACTORED = False
AGENT_NET = GRU_Network

# Pathfinding_Env
NUM_PATTERNS = 7
ONE_HOT_PATTERNS = False
ALLOW_DELIBERATION = False
USE_SUCCESS_RATE = False
HELDOUT_TESTING = False
HP_TUNING_METRIC = MeanReward

###  HYPERPARAMETERS  (tunable)  ###

# GRU_Network ##

# Agents in general
AC_HIDDEN_LAYER_SIZE = 512
BPTT_PERIOD = 16
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.5
GRADIENT_CLIP = 4.0
ENTROPY_REG = 0.02
ADAM_EPS = 1e-08
OBS_FEEDBACK = False

# A3cAgent
REWARD_SCALE = 0.5
WEIGHT_DECAY = 0.
APPLY_RELU_TO_OBS_EMBED = False

# RNNs
NUM_RNN_UNITS = 384
OBS_EMBED_SIZE = 256
