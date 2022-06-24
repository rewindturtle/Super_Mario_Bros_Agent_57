import numpy as np
from numpy import exp, log


FRAME_HEIGHT = 81
FRAME_WIDTH = 81
FRAME_SKIP = 12
NUM_ACTIONS = 10


NUM_PLAYERS = 8
MAX_EPSILON = 1. / 50.
MIN_EPSILON = 1. / 1000.


NUM_ARMS = 32
ARM_EPSILON = 0.5
ARM_BETA = 1.
ARM_WINDOW = 5 * NUM_ARMS
MAX_BETA = 0.3
BETAS = []
for i in range(NUM_ARMS):
    if i == 0:
        BETAS.append(0.)
    elif i == (NUM_ARMS - 1):
        BETAS.append(MAX_BETA)
    else:
        x = 10 * (2 * i - NUM_ARMS + 2) / (NUM_ARMS - 2)
        b = MAX_BETA / (1. + exp(-x))
        BETAS.append(b)


MAX_GAMMA = 0.95
MIN_GAMMA = 0.85
GAMMAS = []
for i in range(NUM_ARMS):
    x = ((NUM_ARMS - i - 1) * log(1 - MAX_GAMMA) + i * log(1 - MIN_GAMMA)) / (NUM_ARMS - 1)
    g = 1 - exp(x)
    GAMMAS.append(g)


UPDATE_TARGET_PERIOD = 100
SAVE_MODEL_PERIOD = 1000
SAVE_DATA_PERIOD = 100
PLAYER_WARM_UP = 100


TRACE_LENGTH = 80
REPLAY_PERIOD = TRACE_LENGTH // 2
RETRACE = 1.
NUM_DENSE = 64
NUM_HASH = NUM_DENSE // 2
SQUISH = 0.01
Q_CLIP = 100


EXTRINSIC_REWARD_NORM = 18.
INTRINSIC_REWARD_NORM = NUM_HASH / 2
PER_ETA = 0.9
PER_EPSILON = 1e-5


LR = 1e-5
INTRINSIC_LR = 5 * LR
CLIP_NORM = 50.
BATCH_SIZE = 64
MAX_MEMORY = 10000
WARM_UP = 10 * BATCH_SIZE # min(150 * BATCH_SIZE, MAX_MEMORY // 3)
CLUSTER = 0.25
HASH_SIZE = BATCH_SIZE * TRACE_LENGTH


PRINT_SUMMARY = False
PRINT_GAME_DATA = True
RENDER = True


MASK_1 = np.zeros((TRACE_LENGTH - 1, TRACE_LENGTH - 1), dtype = np.float32)
for i in range(TRACE_LENGTH - 1):
    for j in range(i, TRACE_LENGTH - 1):
        MASK_1[i, j] = 1.
MASK_1 = np.repeat(MASK_1[None, ...], BATCH_SIZE, axis = 0)
MASK_2 = 1. - MASK_1.copy()
I, J = np.ogrid[:BATCH_SIZE, :TRACE_LENGTH]