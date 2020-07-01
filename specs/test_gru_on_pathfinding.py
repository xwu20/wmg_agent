# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = test  # train, test, test_episodes, render
LOAD_MODEL_FROM = models/gru_pathfinding.pth
SAVE_MODELS_TO = None

# worker.py
AGENT = A3cAgent
ENV = Pathfinding_Env
ENV_MAJOR_RANDOM_SEED = 1
ENV_MINOR_RANDOM_SEED = 0
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 120  # 1200 120
TOTAL_STEPS = 1200  # 12000 1200
ANNEAL_LR = False
LR_GAMMA = 0.98

# A3cAgent
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
