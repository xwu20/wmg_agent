# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_gru_flat_babyai.pth

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
ENV_MAJOR_RANDOM_SEED = 1  # Use randint for non-deterministic behavior.
ENV_MINOR_RANDOM_SEED = 0
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 10000  # 10000 50
TOTAL_STEPS = 2500000  # 2000000 250
ANNEAL_LR = False
LR_GAMMA = 0.98

# A3cAgent
AGENT_NET = GRU_Network

# BabyAI_Env
BABYAI_ENV_LEVEL = BabyAI-GoToLocal-v0
USE_SUCCESS_RATE = True
SUCCESS_RATE_THRESHOLD = 0.99
HELDOUT_TESTING = True
NUM_TEST_EPISODES = 10000
OBS_ENCODER = Flat
BINARY_REWARD = True

# TrajectoryFormatter
USE_TRAJECTORY_FORMATTER = False

###  HYPERPARAMETERS  (tunable)  ###

# GRU_Network ##

# Agents in general
AC_HIDDEN_LAYER_SIZE = 4096
BPTT_PERIOD = 4
LEARNING_RATE = 4e-05
DISCOUNT_FACTOR = 0.9
GRADIENT_CLIP = 512.0
ENTROPY_REG = 0.02
ADAM_EPS = 1e-12
OBS_FEEDBACK = False

# A3cAgent
REWARD_SCALE = 2.0
WEIGHT_DECAY = 0.
APPLY_RELU_TO_OBS_EMBED = False

# RNNs
NUM_RNN_UNITS = 96
OBS_EMBED_SIZE = 512
