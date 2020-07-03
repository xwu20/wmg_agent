# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_wmg_flat_babyai.pth

# worker.py
ENV = BabyAI_Env
ENV_MAJOR_RANDOM_SEED = 2  # Use randint for non-deterministic behavior.
ENV_MINOR_RANDOM_SEED = 0
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 20  # 10000 20
TOTAL_STEPS = 100  # 1600000 100
ANNEAL_LR = False
LR_GAMMA = 0.98

# A3cAgent
AGENT_NET = WMG_Network

# WMG
V2 = False

# BabyAI_Env
BABYAI_ENV_LEVEL = BabyAI-GoToLocal-v0
USE_SUCCESS_RATE = True
SUCCESS_RATE_THRESHOLD = 0.99
HELDOUT_TESTING = True
NUM_TEST_EPISODES = 10000
OBS_ENCODER = Flat
BINARY_REWARD = True

###  HYPERPARAMETERS  (tunable)  ###

# WMG_Network ##

# Agents in general
AC_HIDDEN_LAYER_SIZE = 512
BPTT_PERIOD = 6
LEARNING_RATE = 2.5e-05
DISCOUNT_FACTOR = 0.5
GRADIENT_CLIP = 256.0
ENTROPY_REG = 0.02
ADAM_EPS = 1e-8
OBS_FEEDBACK = False

# A3cAgent
REWARD_SCALE = 16.0
WEIGHT_DECAY = 0.
APPLY_RELU_TO_OBS_EMBED = False

# Transformers
TFM_NUM_LAYERS = 2
TFM_NUM_ATTENTION_HEADS = 16
TFM_ATTENTION_HEAD_SIZE = 24
TFM_HIDDEN_SIZE = 16
TFM_INITIALIZER_RANGE = None
TFM_OUTPUT_ALL_NODES = False

# WMG
WMG_MAX_OBS = 0
WMG_MAX_MEMOS = 16
WMG_MEMO_SIZE = 64
