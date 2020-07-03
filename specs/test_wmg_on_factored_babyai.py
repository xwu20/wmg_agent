# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = test_episodes  # train, test, test_episodes, render
NUM_EPISODES_TO_TEST = 100  # 1000 100
MIN_FINAL_REWARD_FOR_SUCCESS = 1.0
LOAD_MODEL_FROM = models/wmg_factored_babyai.pth
SAVE_MODELS_TO = None

# worker.py
ENV = BabyAI_Env
ENV_MAJOR_RANDOM_SEED = 1
ENV_MINOR_RANDOM_SEED = 0
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 1
TOTAL_STEPS = 1
ANNEAL_LR = False
LR_GAMMA = 0.98

# A3cAgent
REFACTORED = False
AGENT_NET = WMG_Network

# BabyAI_Env
BABYAI_ENV_LEVEL = BabyAI-GoToLocal-v0
USE_SUCCESS_RATE = True
SUCCESS_RATE_THRESHOLD = 0.99
HELDOUT_TESTING = False
NUM_TEST_EPISODES = 10000
OBS_ENCODER = Factored
BINARY_REWARD = True

###  HYPERPARAMETERS  (tunable)  ###

# WMG_Network ##

# Agents in general
AC_HIDDEN_LAYER_SIZE = 2048
BPTT_PERIOD = 6
LEARNING_RATE = 6.3e-05
DISCOUNT_FACTOR = 0.5
GRADIENT_CLIP = 512.0
ENTROPY_REG = 0.1
ADAM_EPS = 1e-12
OBS_FEEDBACK = False

# A3cAgent
REWARD_SCALE = 32.0
WEIGHT_DECAY = 0.
APPLY_RELU_TO_OBS_EMBED = False

# Transformers
TFM_NUM_LAYERS = 4
TFM_NUM_ATTENTION_HEADS = 2
TFM_ATTENTION_HEAD_SIZE = 128
TFM_HIDDEN_SIZE = 32
TFM_INITIALIZER_RANGE = None
TFM_OUTPUT_ALL_NODES = False

# WMG
WMG_MAX_OBS = 0
WMG_MAX_MEMOS = 8
WMG_MEMO_SIZE = 32
