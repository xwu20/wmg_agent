# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_nr_wmg_pathfinding.pth

# worker.py
ENV = Pathfinding_Env
ENV_RANDOM_SEED = 1  # Use randint for non-deterministic behavior.
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 120  # 100000 120
TOTAL_STEPS = 600  # 20000000 600
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
DISCOUNT_FACTOR = 0.6
GRADIENT_CLIP = 16.0
ENTROPY_TERM_STRENGTH = 0.005
ADAM_EPS = 1e-08
OBS_FEEDBACK = False

# A3cAgent
REWARD_SCALE = 1.0
WEIGHT_DECAY = 0.

# WMG
WMG_MAX_OBS = 11
WMG_MAX_MEMOS = 0
WMG_NUM_LAYERS = 4
WMG_NUM_ATTENTION_HEADS = 6
WMG_ATTENTION_HEAD_SIZE = 16
WMG_HIDDEN_SIZE = 32
AC_HIDDEN_LAYER_SIZE = 128
