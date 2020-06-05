import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras

from config import Config
from data_utils import RobertaDataGenerator
from models.roberta import get_roberta
from test import predict_test
from utils import get_jaccard_from_df


def train():
    max_l = Config.Train.max_len
    train_df = pd.read_csv(Config.train_path).dropna()
    val_df = pd.read_csv(Config.validation_path).dropna()
    jaccards = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.seed)
    for i, (train_idx, val_idx) in enumerate(skf.split(train_df.text.values, train_df.sentiment.values)):
        print('=' * 50)
        print(f'# Fold {i + 1}')
        print('=' * 50)

        keras.backend.clear_session()

        _train_generator = RobertaDataGenerator(train_df.iloc[train_idx], augment=False)
        train_dataset = tf.data.Dataset.from_generator(_train_generator.generate,
                                                       output_types=(
                                                           {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                           {'sts': tf.int32, 'ets': tf.int32}))
        train_dataset = train_dataset.padded_batch(Config.Train.batch_size,
                                                   padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                                  {'sts': [max_l], 'ets': [max_l]}),
                                                   padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                                   {'sts': 0, 'ets': 0}))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        _oof_generator = RobertaDataGenerator(train_df.iloc[val_idx], augment=False)
        oof_dataset = tf.data.Dataset.from_generator(_oof_generator.generate,
                                                     output_types=({'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                                   {'sts': tf.int32, 'ets': tf.int32}))
        oof_dataset = oof_dataset.padded_batch(Config.Train.batch_size,
                                               padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                              {'sts': [max_l], 'ets': [max_l]}),
                                               padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                               {'sts': 0, 'ets': 0}))
        oof_dataset = oof_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Training with roberta as feature extractor
        model = get_roberta()
        model.get_layer('tf_roberta_model').trainable = False
        optimizer = keras.optimizers.Adam()
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=Config.Train.label_smoothing)
        model.compile(loss=loss, optimizer=optimizer)
        model.summary()

        cbs = [
            keras.callbacks.EarlyStopping(patience=2, verbose=1, restore_best_weights=True)
        ]
        model.fit(train_dataset, epochs=50, verbose=1, validation_data=oof_dataset, callbacks=cbs)

        # Fine tuning roberta
        model.get_layer('tf_roberta_model').trainable = True
        optimizer = keras.optimizers.Adam(learning_rate=3e-5)
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=Config.Train.label_smoothing)
        model.compile(loss=loss, optimizer=optimizer)
        model.summary()
        cbs = [
            keras.callbacks.ReduceLROnPlateau(patience=2, verbose=1, factor=0.3),
            keras.callbacks.EarlyStopping(patience=3, verbose=1, restore_best_weights=True),
            keras.callbacks.TensorBoard(log_dir=str(Config.Train.tf_log_dir / Config.model_type), histogram_freq=2,
                                        profile_batch=0, write_images=True),
            keras.callbacks.ModelCheckpoint(
                str(Config.Train.checkpoint_dir / Config.model_type / f'weights_{Config.version}_{i}.h5'),
                verbose=1, save_best_only=True, save_weights_only=True)
        ]
        model.fit(train_dataset, epochs=50, verbose=1, validation_data=oof_dataset, callbacks=cbs)

        print('\nLoading model weights...')
        model.load_weights(str(Config.Train.checkpoint_dir / Config.model_type / f'weights_{Config.version}_{i}.h5'))

        print('\nPredicting OOF')
        start_idx, end_idx = model.predict(oof_dataset, verbose=1)
        start_idx = np.argmax(start_idx, axis=-1)
        end_idx = np.argmax(end_idx, axis=-1)
        end_idx = np.where(start_idx > end_idx, start_idx, end_idx)
        jaccard_score = get_jaccard_from_df(train_df.iloc[val_idx], start_idx, end_idx)
        jaccards.append(jaccard_score)
        print(f'\n>> FOLD {i + 1} jaccard score: {jaccard_score}\n')
    print(f'\n>> Mean jaccard score for all {skf.n_splits} folds: {np.mean(jaccards)}\n')
    print('\nPredicting on validation set')
    _val_generator = RobertaDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.Train.batch_size,
                                           padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                          {'sts': [max_l], 'ets': [max_l]}),
                                           padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                           {'sts': 0, 'ets': 0}))
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # noinspection PyUnboundLocalVariable
    start_idx, end_idx = model.predict(val_dataset, verbose=1)
    start_idx = np.argmax(start_idx, axis=-1)
    end_idx = np.argmax(end_idx, axis=-1)
    start_gt_end_df = val_df[start_idx > end_idx]
    start_gt_end_df.to_csv('start_gt_end.csv', index=False)
    end_idx = np.where(start_idx > end_idx, start_idx, end_idx)
    jaccard_score = get_jaccard_from_df(val_df, start_idx, end_idx)
    print(f'\n>> Jaccard score on validation data: {jaccard_score}')
    predict_test()


if __name__ == '__main__':
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    tf.random.set_seed(Config.seed)
    warnings.filterwarnings('ignore')

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    train()
