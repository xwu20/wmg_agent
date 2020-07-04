# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = test  # train, test, test_episodes, render
LOAD_MODEL_FROM = models/wmg_pathfinding.pth
SAVE_MODELS_TO = None

# worker.py
ENV = Pathfinding_Env
ENV_RANDOM_SEED = 1
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 120  # 1200 120
TOTAL_STEPS = 1200  # 12000 1200
ANNEAL_LR = False
LR_GAMMA = 0.98

# A3cAgent
AGENT_NET = WMG_Network

# WMG
V2 = False

# Pathfinding_Env
NUM_PATTERNS = 7
ONE_HOT_PATTERNS = False
ALLOW_DELIBERATION = False
USE_SUCCESS_RATE = False
HELDOUT_TESTING = False

###  HYPERPARAMETERS  (tunable)  ###

# WMG_Network ##

# Agents in general
A3C_T_MAX = 16
LEARNING_RATE = 0.00016
DISCOUNT_FACTOR = 0.5
GRADIENT_CLIP = 16.0
ENTROPY_TERM_STRENGTH = 0.01
ADAM_EPS = 1e-06
OBS_FEEDBACK = False

# A3cAgent
REWARD_SCALE = 2.0
WEIGHT_DECAY = 0.

# WMG
WMG_MAX_OBS = 0
WMG_MAX_MEMOS = 16
WMG_MEMO_SIZE = 128
WMG_NUM_LAYERS = 4
WMG_NUM_ATTENTION_HEADS = 6
WMG_ATTENTION_HEAD_SIZE = 12
WMG_HIDDEN_SIZE = 12
AC_HIDDEN_LAYER_SIZE = 128
