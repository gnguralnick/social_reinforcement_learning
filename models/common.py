from tensorflow.python.keras import layers


def build_q_arch(state_input, pos_input):
    flat_state = layers.Flatten()(state_input)
    state1 = layers.Dense(512, activation="relu")(flat_state)
    flat_pos = layers.Flatten()(pos_input)
    pos1 = layers.Dense(16, activation="linear")(flat_pos)
    layer3 = layers.Concatenate()([pos1, state1])
    layer4 = layers.Dense(256, activation="relu")(layer3)

    return layer4
