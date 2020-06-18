# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_model_2.pth

# rl_train.py
USE_DGD = False
USE_DGD2 = False
NUM_HPS = 15
ARCHIVE_ALL_MODELS = False
XT_LOAD_MODEL = False
XT_LOAD_MODEL_WS = ws1133
XT_LOAD_MODEL_STEP = 1000000

# worker.py
AGENT = AacAgent
ENV = Sokoban_Env
ENV_MAJOR_RANDOM_SEED = 1
ENV_MINOR_RANDOM_SEED = 0
AGENT_RANDOM_SEED = 2
REPORTING_INTERVAL = 100  # 100000 for full training run.
TOTAL_STEPS = 1000  # 20000000 for full training run.
ANNEAL_LR = False
LR_GAMMA = 0.98

# AacAgent
AGENT_NET = WMG2_Network

# Sokoban_Env
SOKOBAN_MAX_STEPS = 120
SOKOBAN_DIFFICULTY = unfiltered
SOKOBAN_SPLIT = train
SOKOBAN_ROOM_OVERRIDE = None
SOKOBAN_BOXES_REQUIRED = 4
SOKOBAN_OBSERVATION_FORMAT = factored
HP_TUNING_METRIC = FinalSuccessRate

# TrajectoryFormatter
USE_TRAJECTORY_FORMATTER = False

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

# AacAgent
REWARD_SCALE = 4.
WEIGHT_DECAY = 0.
NUM_AC_LAYERS = 1

# Transformers
TFM_NUM_LAYERS = 10
TFM_NUM_ATTENTION_HEADS = 8
TFM_ATTENTION_HEAD_SIZE = 32
TFM_HIDDEN_SIZE = 8

# WMG2
WMG2_MAX_CONCEPTS = 1
WMG2_CONCEPT_SIZE = 2048
WMG2_OUTPUT_LAYER_SIZE = 2880
WMG2_MAX_AGE = 1
