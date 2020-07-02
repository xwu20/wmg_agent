# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_nr_wmg_pathfinding.pth

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
AGENT_NET = WMG_Network
WMG_HIST_MEMS = True

# Pathfinding_Env
NUM_PATTERNS = 7
ONE_HOT_PATTERNS = False
ALLOW_DELIBERATION = False
USE_SUCCESS_RATE = False
HELDOUT_TESTING = False
HP_TUNING_METRIC = MeanReward

###  HYPERPARAMETERS  (tunable)  ###

# WMG_Network ##

# Agents in general
AC_HIDDEN_LAYER_SIZE = 128
BPTT_PERIOD = 16
LEARNING_RATE = 0.00016
DISCOUNT_FACTOR = 0.6
GRADIENT_CLIP = 16.0
ENTROPY_REG = 0.005
ADAM_EPS = 1e-08
OBS_FEEDBACK = False

# A3cAgent
REWARD_SCALE = 1.0
WEIGHT_DECAY = 0.
APPLY_RELU_TO_OBS_EMBED = False

# Transformers
TFM_NUM_LAYERS = 4
TFM_NUM_ATTENTION_HEADS = 6
TFM_ATTENTION_HEAD_SIZE = 16
TFM_HIDDEN_SIZE = 32
TFM_INITIALIZER_RANGE = None
TFM_OUTPUT_ALL_NODES = False

# WMG
WMG_MAX_MEMS = 11
