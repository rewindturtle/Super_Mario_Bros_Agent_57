from Hyperparameters import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import tensorflow.keras.backend as K


PRINT_SUMMARY = True


def dueling_output(input_layer):
    state_layer = input_layer[0]
    action_layer = input_layer[1]
    return state_layer + action_layer - K.mean(action_layer, axis = 1, keepdims = True)


def h(input_layer):
    squish = K.sign(input_layer) * (K.sqrt(K.abs(input_layer) + 1) - 1) + SQUISH * input_layer
    return squish


def h_inv(input_layer):
    arg = 4 * SQUISH * (K.abs(input_layer) + SQUISH + 1) + 1
    f1 = (1 - K.sqrt(arg)) / (2 * (SQUISH ** 2))
    f2 = (K.abs(input_layer) + 1) / SQUISH
    stretch = K.sign(input_layer) * (f1 + f2)
    return stretch


def combine_q(input_layer):
    qe = h_inv(input_layer[0])
    qi = h_inv(input_layer[1])
    beta = input_layer[2]
    q = h(qe + beta * qi)
    return q


def create_conv_input():
    frame_input = Input(shape = FRAME_SHAPE)
    conv1 = Conv2D(16,
                   kernel_size = (9, 9),
                   strides = 3,
                   kernel_initializer = 'he_uniform',
                   activation = 'relu')(frame_input)
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


def create_inner_player_predictor():
    frame_input = Input(shape = FRAME_SHAPE)
    action_input = Input(shape = NUM_ACTIONS)
    discount_input = Input(shape = 1)
    conv = create_conv_input()(frame_input)
    conc1 = Concatenate()([conv, action_input])
    conc1_exp = K.expand_dims(conc1, axis = 0)
    lstm = LSTM(NUM_DENSE,
                kernel_initializer = 'he_uniform',
                activation = 'relu',
                stateful = True)(conc1_exp)
    conc2 = Concatenate()([lstm, discount_input])
    dense1 = Dense(NUM_DENSE,
                   kernel_initializer = 'he_uniform',
                   activation = 'relu')(conc2)
    dense2 = Dense(NUM_DENSE,
                   kernel_initializer = 'he_uniform',
                   activation = 'relu')(conc2)
    action_dense = Dense(NUM_ACTIONS,
                         kernel_initializer = 'he_uniform',
                         activation = 'relu')(dense1)
    state_dense = Dense(1,
                        kernel_initializer = 'he_uniform',
                        activation = 'relu')(dense2)
    duel = Lambda(dueling_output)([state_dense, action_dense])
    model = Model([frame_input, action_input, discount_input],
                  duel)
    return model


def create_player_predictor():
    frame_input = Input(shape = FRAME_SHAPE)
    action_input = Input(shape = NUM_ACTIONS)
    discount_input = Input(shape = 1)
    beta_input = Input(shape = 1)
    inner1 = create_inner_player_predictor()([frame_input, action_input, discount_input])
    inner2 = create_inner_player_predictor()([frame_input, action_input, discount_input])
    q_output = Lambda(combine_q)([inner1, inner2, beta_input])
    model = Model([frame_input, action_input, discount_input, beta_input],
                  q_output)
    if PRINT_SUMMARY:
        print(model.summary())
    return model


def create_trainer_predictor():
    frame_input = Input(shape = FRAME_SHAPE)
    action_input = Input(shape = NUM_ACTIONS)
    conv = TimeDistributed(create_conv_input())(frame_input)
    conc = TimeDistributed(Concatenate())([conv, action_input])
    lstm = LSTM(NUM_DENSE,
                kernel_initializer = 'he_uniform',
                activation = 'relu',
                stateful = False)(conc)


if __name__ == '__main__':
    player_predictor = create_player_predictor()