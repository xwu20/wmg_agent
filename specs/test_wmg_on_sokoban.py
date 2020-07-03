# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = test_episodes  # train, test, test_episodes, render
NUM_EPISODES_TO_TEST = 10  # 1000 10
MIN_FINAL_REWARD_FOR_SUCCESS = 2.0
LOAD_MODEL_FROM = models/wmg_sokoban.pth
SAVE_MODELS_TO = None

# worker.py
ENV = Sokoban_Env
ENV_MAJOR_RANDOM_SEED = 1
ENV_MINOR_RANDOM_SEED = 0
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 10000
TOTAL_STEPS = 100000
ANNEAL_LR = False
LR_GAMMA = 0.98

# A3cAgent
REFACTORED = True
AGENT_NET = WMG_Network

# Sokoban_Env
SOKOBAN_MAX_STEPS = 120
SOKOBAN_DIFFICULTY = unfiltered
SOKOBAN_SPLIT = test
SOKOBAN_ROOM_OVERRIDE = None
SOKOBAN_BOXES_REQUIRED = 4
SOKOBAN_OBSERVATION_FORMAT = factored
HP_TUNING_METRIC = FinalSuccessRate

###  HYPERPARAMETERS  (tunable)  ###

# Sokoban_Env
SOKOBAN_REWARD_PER_STEP = 0.
SOKOBAN_REWARD_SUCCESS = 2.

# Agents in general
BPTT_PERIOD = 4
LEARNING_RATE = 1.6e-05
DISCOUNT_FACTOR = 0.995
GRADIENT_CLIP = 512.0
ENTROPY_REG = 0.02
ADAM_EPS = 1e-10

# A3cAgent
REWARD_SCALE = 4.
WEIGHT_DECAY = 0.
NUM_AC_LAYERS = 1

# Transformers
TFM_NUM_LAYERS = 10
TFM_NUM_ATTENTION_HEADS = 8
TFM_ATTENTION_HEAD_SIZE = 32
TFM_HIDDEN_SIZE = 8

# WMG2
WMG_MAX_OBS = 0
WMG_MAX_MEMOS = 1
WMG_MEMO_SIZE = 2048
AC_HIDDEN_LAYER_SIZE = 2880
