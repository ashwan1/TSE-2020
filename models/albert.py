import tensorflow as tf
from tensorflow import keras
from transformers import AlbertConfig, TFAlbertModel

from config import Config


def get_albert():
    ids = keras.layers.Input(shape=(None,), dtype=tf.int32, name='ids')
    att = keras.layers.Input(shape=(None,), dtype=tf.int32, name='att')
    tok_type_ids = keras.layers.Input(shape=(None,), dtype=tf.int32, name='tti')

    config = AlbertConfig.from_pretrained(Config.Albert.config)
    albert_model = TFAlbertModel.from_pretrained(Config.Albert.model, config=config)

    x = albert_model(ids, attention_mask=att, token_type_ids=tok_type_ids)

    x1 = keras.layers.Dropout(0.15)(x[0])
    x1 = keras.layers.Conv1D(768, 2, padding='same')(x1)
    x1 = keras.layers.LeakyReLU()(x1)
    x1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)(x1)
    x1 = keras.layers.Conv1D(64, 2, padding='same')(x1)
    x1 = keras.layers.LeakyReLU()(x1)
    x1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)(x1)
    x1 = keras.layers.Conv1D(32, 2, padding='same')(x1)
    x1 = keras.layers.Conv1D(1, 1)(x1)
    x1 = keras.layers.Flatten()(x1)
    x1 = keras.layers.Activation('softmax', dtype='float32', name='sts')(x1)

    x2 = keras.layers.Dropout(0.15)(x[0])
    x2 = keras.layers.Conv1D(768, 2, padding='same')(x2)
    x2 = keras.layers.LeakyReLU()(x2)
    x2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)(x2)
    x2 = keras.layers.Conv1D(64, 2, padding='same')(x2)
    x2 = keras.layers.LeakyReLU()(x2)
    x2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)(x2)
    x2 = keras.layers.Conv1D(32, 2, padding='same')(x2)
    x2 = keras.layers.Conv1D(1, 1)(x2)
    x2 = keras.layers.Flatten()(x2)
    x2 = keras.layers.Activation('softmax', dtype='float32', name='ets')(x2)

    model = keras.models.Model(inputs=[ids, att, tok_type_ids], outputs=[x1, x2])

    optimizer = keras.optimizers.Adam(learning_rate=6e-5)
    if Config.Train.use_amp:
        optimizer = keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=Config.Train.label_smoothing)
    model.compile(loss=loss, optimizer=optimizer)

    return model
