import tensorflow as tf
from kerastuner import HyperParameters
from tensorflow import keras
from transformers import RobertaConfig, TFRobertaModel

from config import Config


def get_roberta():
    ids = keras.layers.Input(shape=(None,), dtype=tf.int32, name='ids')
    att = keras.layers.Input(shape=(None,), dtype=tf.int32, name='att')
    tok_type_ids = keras.layers.Input(shape=(None,), dtype=tf.int32, name='tti')

    config = RobertaConfig.from_pretrained(Config.Roberta.config)
    roberta_model = TFRobertaModel.from_pretrained(Config.Roberta.model, config=config)

    x = roberta_model(ids, attention_mask=att, token_type_ids=tok_type_ids)

    x1 = keras.layers.Dropout(0.15)(x[0])
    x1 = keras.layers.Conv1D(768, 2, padding='same')(x1)
    x1 = keras.layers.LeakyReLU()(x1)
    x1 = keras.layers.LayerNormalization()(x1)
    x1 = keras.layers.Conv1D(64, 2, padding='same')(x1)
    x1 = keras.layers.LeakyReLU()(x1)
    x1 = keras.layers.LayerNormalization()(x1)
    x1 = keras.layers.Conv1D(32, 2, padding='same')(x1)
    x1 = keras.layers.Dense(1)(x1)
    x1 = keras.layers.Flatten()(x1)
    x1 = keras.layers.Activation('softmax', dtype='float32', name='sts')(x1)

    x2 = keras.layers.Dropout(0.15)(x[0])
    x2 = keras.layers.Conv1D(768, 2, padding='same')(x2)
    x2 = keras.layers.LeakyReLU()(x2)
    x2 = keras.layers.LayerNormalization()(x2)
    x2 = keras.layers.Conv1D(64, 2, padding='same')(x2)
    x2 = keras.layers.LeakyReLU()(x2)
    x2 = keras.layers.LayerNormalization()(x2)
    x2 = keras.layers.Conv1D(32, 2, padding='same')(x2)
    x2 = keras.layers.Dense(1)(x2)
    x2 = keras.layers.Flatten()(x2)
    x2 = keras.layers.Activation('softmax', dtype='float32', name='ets')(x2)

    model = keras.models.Model(inputs=[ids, att, tok_type_ids], outputs=[x1, x2])

    optimizer = keras.optimizers.Adam(learning_rate=6e-5)
    if Config.Train.use_amp:
        optimizer = keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=Config.Train.label_smoothing)
    model.compile(loss=loss, optimizer=optimizer)

    return model


def get_tunable_roberta(hp: HyperParameters):
    ids = keras.layers.Input(shape=(Config.Train.max_len,), dtype=tf.int32, name='ids')
    att = keras.layers.Input(shape=(Config.Train.max_len,), dtype=tf.int32, name='att')
    tok_type_ids = keras.layers.Input(shape=(Config.Train.max_len,), dtype=tf.int32, name='tti')

    config = RobertaConfig.from_pretrained(Config.Roberta.config)
    roberta_model = TFRobertaModel.from_pretrained(Config.Roberta.model, config=config)

    roberta_model.trainable = False

    x = roberta_model(ids, attention_mask=att, token_type_ids=tok_type_ids)

    use_alpha_dropout = False  # hp.Boolean('use_alpha_dropout')
    if use_alpha_dropout:
        x1 = keras.layers.AlphaDropout(hp.Choice('dropout1', [0.1, 0.2, 0.3]))(x[0])
        x2 = keras.layers.AlphaDropout(hp.Choice('dropout2', [0.1, 0.2, 0.3]))(x[0])
    else:
        x1 = keras.layers.Dropout(hp.Choice('dropout1', [0.1, 0.2, 0.3]))(x[0])
        x2 = keras.layers.Dropout(hp.Choice('dropout2', [0.1, 0.2, 0.3]))(x[0])

    use_rnn = False  # hp.Boolean('use_rnn')
    if use_rnn:
        lstm_count = hp.Choice('rnn_count', [1, 2])
        for i in range(lstm_count):
            x1, state1_0, _, state1_1, _ = keras.layers.Bidirectional(
                keras.layers.LSTM(hp.Int(f'lstm_units1_{i}', 32, 48, step=8),
                                  return_sequences=True,
                                  return_state=True))(x1)
            x1 = keras.layers.LeakyReLU()(x1)
            state1 = keras.layers.concatenate([state1_0, state1_1])
            x1 = keras.layers.Attention()([x1, state1])
            x2, state2_0, _, state2_1, _ = keras.layers.Bidirectional(
                keras.layers.LSTM(hp.Int(f'lstm_units2_{i}', 32, 48, step=8),
                                  return_sequences=True,
                                  return_state=True))(x2)
            x2 = keras.layers.LeakyReLU()(x2)
            state2 = keras.layers.concatenate([state2_0, state2_1])
            x2 = keras.layers.Attention()([x2, state2])
    else:
        conv_count = hp.Choice('conv_count', [1, 2])
        for i in range(conv_count):
            x1 = keras.layers.Conv1D(hp.Int(f'conv_filter1_{i}', 8, 24, step=8),
                                     hp.Int(f'conv_kernel1_{i}', 3, 5, step=1), padding='same')(x1)
            x1 = keras.layers.LeakyReLU()(x1)
            x2 = keras.layers.Conv1D(hp.Int(f'conv_filter2_{i}', 8, 24, step=8),
                                     hp.Int(f'conv_kernel2_{i}', 3, 5, step=1), padding='same')(x2)
            x2 = keras.layers.LeakyReLU()(x2)

    x1 = keras.layers.Conv1D(1, 1)(x1)
    x1 = keras.layers.Flatten()(x1)
    x1 = keras.layers.Activation('softmax', name='sts')(x1)

    x2 = keras.layers.Conv1D(1, 1)(x2)
    x2 = keras.layers.Flatten()(x2)
    x2 = keras.layers.Activation('softmax', name='ets')(x2)

    model = keras.models.Model(inputs=[ids, att, tok_type_ids], outputs=[x1, x2])
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=Config.Train.label_smoothing)
    model.compile(loss=loss, optimizer=optimizer)

    return model


def get_classification_roberta():
    ids = keras.layers.Input(shape=(Config.Train.max_len,), dtype=tf.int32, name='ids')
    att = keras.layers.Input(shape=(Config.Train.max_len,), dtype=tf.int32, name='att')
    tok_type_ids = keras.layers.Input(shape=(Config.Train.max_len,), dtype=tf.int32, name='tti')

    config = RobertaConfig.from_pretrained(Config.Roberta.config)
    roberta_model = TFRobertaModel.from_pretrained(Config.Roberta.model, config=config)

    x = roberta_model(ids, attention_mask=att, token_type_ids=tok_type_ids)

    x = keras.layers.Dropout(0.2)(x[0])
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(3, activation='softmax', name='sentiment')(x)

    model = keras.models.Model(inputs=[ids, att, tok_type_ids], outputs=x)
    lr_schedule = keras.experimental.CosineDecay(5e-5, 1000)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=Config.Train.label_smoothing)
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    return model
