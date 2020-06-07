import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config import Config
from data_utils import RobertaDataGenerator
from models.roberta import get_roberta
from utils import get_jaccard_from_df


def train_roberta():
    max_l = Config.Train.max_len
    train_df = pd.read_csv(Config.train_path).dropna()
    val_df = pd.read_csv(Config.validation_path).dropna()
    _train_generator = RobertaDataGenerator(train_df, augment=True)
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

    model = get_roberta()
    model.summary()

    cbs = [
        # keras.callbacks.ReduceLROnPlateau(patience=2, verbose=1, factor=0.3),
        keras.callbacks.EarlyStopping(patience=3, verbose=1, restore_best_weights=True),
        keras.callbacks.TensorBoard(log_dir=str(Config.Train.tf_log_dir / Config.model_type), histogram_freq=2,
                                    profile_batch=0, write_images=True),
        keras.callbacks.ModelCheckpoint(
            str(Config.Train.checkpoint_dir / Config.model_type / f'weights_{Config.version}.h5'),
            verbose=1, save_best_only=True, save_weights_only=True)
    ]
    model.fit(train_dataset, epochs=50, verbose=1, validation_data=val_dataset, callbacks=cbs)

    print('\nLoading model weights...')
    model.load_weights(str(Config.Train.checkpoint_dir / Config.model_type / f'weights_{Config.version}.h5'))

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
    start_idx, end_idx = model.predict(val_dataset, verbose=1)
    start_idx = np.argmax(start_idx, axis=-1)
    end_idx = np.argmax(end_idx, axis=-1)
    val_df = val_df[_val_generator.exception_mask]
    start_gt_end_df = val_df[start_idx > end_idx]
    start_gt_end_df.to_csv('start_gt_end.csv', index=False)
    end_idx = np.where(start_idx > end_idx, start_idx, end_idx)
    jaccard_score = get_jaccard_from_df(val_df, start_idx, end_idx)
    print(f'\n>> Jaccard score on validation data: {jaccard_score}')
    # predict_test()


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
    if Config.model_type == 'roberta' or Config.model_type == 'distill_roberta':
        train_roberta()
