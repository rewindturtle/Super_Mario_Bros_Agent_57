from Hyperparameters import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow import convert_to_tensor, repeat, one_hot, int64, gather


def dueling_output(input_layer):
    state_layer = input_layer[0]
    action_layer = input_layer[1]
    return state_layer + action_layer - K.mean(action_layer, axis = 1, keepdims = True)


def h(input_layer):
    squish = K.sign(input_layer) * (K.sqrt(K.abs(input_layer) + 1) - 1) + SQUISH * input_layer
    return squish


def h_inv(input_layer):
    input_layer = K.clip(input_layer, -Q_CLIP, Q_CLIP)
    arg = 4 * SQUISH * (K.abs(input_layer) + SQUISH + 1) + 1
    f1 = (1 - K.sqrt(arg)) / (2 * (SQUISH ** 2))
    f2 = (K.abs(input_layer) + 1) / SQUISH
    stretch = K.sign(input_layer) * (f1 + f2)
    return stretch


def combine_q(input_layer):
    qe = h_inv(input_layer[0])
    qi = h_inv(input_layer[1])
    betas = convert_to_tensor(BETAS)
    arm_hot = K.squeeze(one_hot(input_layer[2], NUM_ARMS), axis = 1)
    beta = K.sum(betas * arm_hot)
    q = h(qe + beta * qi)
    return q


def create_conv_input():
    frame_input = Input(shape = (FRAME_HEIGHT, FRAME_WIDTH))
    frame_exp = K.expand_dims(frame_input, axis = -1)
    conv1 = Conv2D(16,
                   kernel_size = (9, 9),
                   strides = 3,
                   kernel_initializer = 'he_uniform',
                   activation = 'relu')(frame_exp)
    conv2 = Conv2D(32,
                   kernel_size = (7, 7),
                   strides = 2,
                   kernel_initializer = 'he_uniform',
                   activation = 'relu')(conv1)
    conv3 = Conv2D(32,
                   kernel_size = (5, 5),
                   strides = 1,
                   kernel_initializer = 'he_uniform',
                   activation = 'relu')(conv2)
    flat = Flatten()(conv3)
    model = Model(frame_input, flat)
    return model


def create_time_dist_conv_input():
    frame_input = Input(shape = (None, FRAME_HEIGHT, FRAME_WIDTH))
    frame_exp = K.expand_dims(frame_input, axis = -1)
    conv1 = TimeDistributed(Conv2D(16,
                                   kernel_size = (9, 9),
                                   strides = 3,
                                   kernel_initializer = 'he_uniform',
                                   activation = 'relu'))(frame_exp)
    conv2 = TimeDistributed(Conv2D(32,
                                   kernel_size = (7, 7),
                                   strides = 2,
                                   kernel_initializer = 'he_uniform',
                                   activation = 'relu'))(conv1)
    conv3 = TimeDistributed(Conv2D(32,
                                   kernel_size = (5, 5),
                                   strides = 1,
                                   kernel_initializer = 'he_uniform',
                                   activation = 'relu'))(conv2)
    flat = TimeDistributed(Flatten())(conv3)
    model = Model(frame_input, flat)
    return model


def create_inner_player_predictor():
    frame_input = Input(shape = (FRAME_HEIGHT, FRAME_WIDTH), batch_size = 1)
    past_action_input = Input(shape = 1, dtype = int64, batch_size = 1)
    arm_input = Input(shape = 1, dtype = int64, batch_size = 1)

    conv = create_conv_input()(frame_input)
    a_hot = K.squeeze(one_hot(past_action_input, NUM_ACTIONS), axis = 1)
    arm_hot = K.squeeze(one_hot(arm_input, NUM_ARMS), axis = 1)

    conc_1 = Concatenate()([conv, a_hot])
    conc1_exp = K.expand_dims(conc_1, axis = 1)
    lstm = LSTM(NUM_DENSE,
                kernel_initializer = 'he_uniform',
                activation = 'relu',
                stateful = True)(conc1_exp)

    conc_2 = Concatenate()([lstm, arm_hot])
    dense_1 = Dense(NUM_DENSE,
                    kernel_initializer = 'he_uniform',
                    activation = 'relu')(conc_2)
    dense_2 = Dense(NUM_DENSE,
                    kernel_initializer = 'he_uniform',
                    activation = 'relu')(conc_2)
    action_dense = Dense(NUM_ACTIONS,
                         kernel_initializer = 'he_uniform',
                         activation = 'relu')(dense_1)
    state_dense = Dense(1,
                        kernel_initializer = 'he_uniform',
                        activation = 'relu')(dense_2)
    duel = Lambda(dueling_output)([state_dense, action_dense])
    model = Model([frame_input, past_action_input, arm_input],
                  duel)
    if PRINT_SUMMARY:
        print(model.summary())
    return model


def create_player_predictor():
    frame_input = Input(shape = (FRAME_HEIGHT, FRAME_WIDTH))
    past_action_input = Input(shape = 1, dtype = int64)
    arm_input = Input(shape = 1, dtype = int64)

    inner1 = create_inner_player_predictor()([frame_input, past_action_input, arm_input])
    inner2 = create_inner_player_predictor()([frame_input, past_action_input, arm_input])
    q = Lambda(combine_q)([inner1, inner2, arm_input])
    model = Model([frame_input, past_action_input, arm_input],
                  q)
    if PRINT_SUMMARY:
        print(model.summary())
    return model


def create_inner_trainer_predictor():
    frame_input = Input(shape = (TRACE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH))
    past_action_input = Input(shape = TRACE_LENGTH, dtype = int64)
    arm_input = Input(shape = 1, dtype = int64)

    conv = create_time_dist_conv_input()(frame_input)
    a_hot = one_hot(past_action_input, NUM_ACTIONS)
    arm_hot = one_hot(arm_input, NUM_ARMS)

    conc_1 = TimeDistributed(Concatenate())([conv, a_hot])
    lstm = LSTM(NUM_DENSE,
                kernel_initializer = 'he_uniform',
                activation = 'relu',
                stateful = False,
                return_sequences = True)(conc_1)

    #arm_exp = K.expand_dims(arm_hot, axis = 0)
    arm_repeat = repeat(arm_hot, TRACE_LENGTH, axis = 0)


    conc_2 = TimeDistributed(Concatenate())([lstm, arm_repeat])
    dense_1 = TimeDistributed(Dense(NUM_DENSE,
                                    kernel_initializer = 'he_uniform',
                                    activation = 'relu'))(conc_2)
    dense_2 = TimeDistributed(Dense(NUM_DENSE,
                                    kernel_initializer='he_uniform',
                                    activation='relu'))(conc_2)
    action_dense = TimeDistributed(Dense(NUM_ACTIONS,
                                         kernel_initializer='he_uniform',
                                         activation='relu'))(dense_1)
    state_dense = TimeDistributed(Dense(1,
                                        kernel_initializer = 'he_uniform',
                                        activation = 'relu'))(dense_2)
    duel = TimeDistributed(Lambda(dueling_output))([state_dense, action_dense])
    model = Model([frame_input, past_action_input, arm_input],
                  duel)
    if PRINT_SUMMARY:
        print(model.summary())
    return model


def create_trainer_predictor():
    frame_input = Input(shape = (TRACE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH))
    past_action_input = Input(shape = TRACE_LENGTH, dtype = int64)
    arm_input = Input(shape = 1, dtype = int64)

    inner1 = create_inner_trainer_predictor()([frame_input, past_action_input, arm_input])
    inner2 = create_inner_trainer_predictor()([frame_input, past_action_input, arm_input])
    model = Model([frame_input, past_action_input, arm_input],
                  [inner1, inner2])
    model.compile(optimizer = Adam(learning_rate = LR, clipnorm = CLIP_NORM),
                  loss = [Huber(), Huber()])
    if PRINT_SUMMARY:
        print(model.summary())
    return model


def create_target_predictor():
    frame_input = Input(shape = (TRACE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH))
    past_action_input = Input(shape = TRACE_LENGTH, dtype = int64)
    arm_input = Input(shape = 1, dtype = int64)

    inner1 = create_inner_trainer_predictor()([frame_input, past_action_input, arm_input])
    inner2 = create_inner_trainer_predictor()([frame_input, past_action_input, arm_input])
    model = Model([frame_input, past_action_input, arm_input],
                  [inner1, inner2])
    if PRINT_SUMMARY:
        print(model.summary())
    return model


def create_player_hasher():
    frame_input = Input(shape = (FRAME_HEIGHT, FRAME_WIDTH))
    conv = create_conv_input()(frame_input)
    dense = Dense(NUM_DENSE // 4,
                  kernel_initializer = 'he_uniform',
                  activation = 'relu')(conv)
    model = Model(frame_input,
                  dense)
    if PRINT_SUMMARY:
        print(model.summary())
    return model


def create_trainer_hasher():
    frame_input1 = Input(shape = (FRAME_HEIGHT, FRAME_WIDTH))
    frame_input2 = Input(shape = (FRAME_HEIGHT, FRAME_WIDTH))
    hasher = create_player_hasher()(frame_input1, frame_input2)
    dense1 = Dense(NUM_DENSE // 2,
                   kernel_initializer = 'he_uniform',
                   activation = 'relu')(hasher)
    dense2 = Dense(NUM_ACTIONS,
                   kernel_initializer = 'he_uniform')(dense1)
    softmax = Softmax()(dense2)
    model = Model([frame_input1, frame_input2],
                  softmax)
    model.compile(optimizer = Adam(learning_rate = RND_LR, clipnorm = CLIP_NORM),
                  loss = Huber())
    if PRINT_SUMMARY:
        print(model.summary())
    return model


def create_player_rnd():
    frame_input = Input(shape = (FRAME_HEIGHT, FRAME_WIDTH))
    conv = create_conv_input()(frame_input)
    dense = Dense(NUM_DENSE // 2,
                  kernel_initializer = 'he_uniform')(conv)
    model = Model(frame_input,
                  dense)
    if PRINT_SUMMARY:
        print(model.summary())
    return model


def create_target_rnd():
    frame_input = Input(shape = (FRAME_HEIGHT, FRAME_WIDTH))
    frame_exp = K.expand_dims(frame_input, axis = -1)
    conv1 = Conv2D(16,
                   kernel_size = (9, 9),
                   strides = 3,
                   kernel_initializer = 'random_uniform',
                   bias_initializer = 'random_uniform',
                   activation = 'relu')(frame_exp)
    conv2 = Conv2D(32,
                   kernel_size = (7, 7),
                   strides = 2,
                   kernel_initializer = 'random_uniform',
                   bias_initializer = 'random_uniform',
                   activation = 'relu')(conv1)
    conv3 = Conv2D(32,
                   kernel_size = (5, 5),
                   strides = 1,
                   kernel_initializer = 'random_uniform',
                   bias_initializer = 'random_uniform',
                   activation = 'relu')(conv2)
    flat = Flatten()(conv3)
    dense = Dense(NUM_DENSE // 2,
                  kernel_initializer = 'random_uniform',
                  bias_initializer = 'random_uniform',)(flat)
    model = Model(frame_input,
                  dense)
    if PRINT_SUMMARY:
        print(model.summary())
    return model


def create_trainer_rnd():
    frame_input = Input(shape = (FRAME_HEIGHT, FRAME_WIDTH))
    conv = create_conv_input()(frame_input)
    dense = Dense(NUM_DENSE // 2,
                  kernel_initializer = 'he_uniform')(conv)
    model = Model(frame_input,
                  dense)
    model.compile(optimizer = Adam(learning_rate = RND_LR, clipnorm = CLIP_NORM),
                  loss = Huber())
    if PRINT_SUMMARY:
        print(model.summary())
    return model