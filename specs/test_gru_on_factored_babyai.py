# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = test_episodes  # train, test, test_episodes, render
NUM_EPISODES_TO_TEST = 1000
MIN_FINAL_REWARD_FOR_SUCCESS = 1.0
LOAD_MODEL_FROM = models/gru_factored_babyai.pth
SAVE_MODELS_TO = None

# rl_train.py
USE_DGD = False
USE_DGD2 = False
NUM_HPS = 15
ARCHIVE_ALL_MODELS = False
XT_LOAD_MODEL = False
XT_LOAD_MODEL_WS = ws1133
XT_LOAD_MODEL_STEP = 1000000

# worker.py
AGENT = A3cAgent
ENV = BabyAI_Env
ENV_MAJOR_RANDOM_SEED = 1
ENV_MINOR_RANDOM_SEED = 0
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 1
TOTAL_STEPS = 1
ANNEAL_LR = False
LR_GAMMA = 0.98

# A3cAgent
AGENT_NET = GRU_Network

# BabyAI_Env
BABYAI_ENV_LEVEL = BabyAI-GoToLocal-v0
USE_SUCCESS_RATE = True
SUCCESS_RATE_THRESHOLD = 0.99
HELDOUT_TESTING = False
NUM_TEST_EPISODES = 10000
OBS_ENCODER = FactoredThenFlattened
BINARY_REWARD = True

# TrajectoryFormatter
USE_TRAJECTORY_FORMATTER = False

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
