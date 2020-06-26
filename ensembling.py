import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path

import pandas as pd
import numpy as np

import tensorflow as tf

from config import Config
from data_utils import RobertaDataGenerator
from models import get_roberta
from utils import get_jaccard_from_df


def base_ensemble():
    max_l = Config.Train.max_len
    val_df = pd.read_csv(Config.validation_path)

    models_paths = list(Path(Config.Train.checkpoint_dir / Config.model_type).iterdir())
    start_idx = 0
    end_idx = 0
    jaccards = []
    for path in models_paths:
        tf.keras.backend.clear_session()

        _generator = RobertaDataGenerator(val_df, augment=False)
        dataset = tf.data.Dataset.from_generator(_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
        dataset = dataset.padded_batch(Config.Train.batch_size,
                                       padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                      {'sts': [max_l], 'ets': [max_l]}),
                                       padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                       {'sts': 0, 'ets': 0}))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        model = get_roberta()
        model.load_weights(str(path))

        s_idx, e_idx = model.predict(dataset, verbose=1)
        start_idx += s_idx
        end_idx += e_idx
        jaccard = get_jaccard_from_df(val_df, np.argmax(s_idx, axis=-1),
                                      np.argmax(e_idx, axis=-1), 'roberta', None)
        jaccards.append(jaccard)
    print(f'\nMean jaccard for all models: {np.mean(jaccards)}')
    start_idx /= 5
    end_idx /= 5
    e_jaccard = get_jaccard_from_df(val_df, np.argmax(start_idx, axis=-1),
                                    np.argmax(end_idx, axis=-1), 'roberta', None)
    print(f'Mean ensemble jaccard for models (base): {e_jaccard}\n')


def max_joint_proba_ensemble():
    max_l = Config.Train.max_len
    val_df = pd.read_csv(Config.validation_path)
    joint_probs = np.zeros((val_df.shape[0], 5))
    start_idx = np.zeros((val_df.shape[0], 5))
    end_idx = np.zeros((val_df.shape[0], 5))
    models_paths = list(Path(Config.Train.checkpoint_dir / Config.model_type).iterdir())
    for i, path in enumerate(models_paths):
        tf.keras.backend.clear_session()

        _generator = RobertaDataGenerator(val_df, augment=False)
        dataset = tf.data.Dataset.from_generator(_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
        dataset = dataset.padded_batch(Config.Train.batch_size,
                                       padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                      {'sts': [max_l], 'ets': [max_l]}),
                                       padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                       {'sts': 0, 'ets': 0}))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        model = get_roberta()
        model.load_weights(str(path))

        s_idx, e_idx = model.predict(dataset, verbose=1)
        joint_probs[:, i] = np.max(s_idx, axis=-1) * np.max(e_idx, axis=-1)
        start_idx[:, i] = np.argmax(s_idx, axis=-1)
        end_idx[:, i] = np.argmax(e_idx, axis=-1)
    selection_idx = np.argmax(joint_probs, axis=-1)
    start_idx = start_idx[:, selection_idx][:, 0]
    end_idx = end_idx[:, selection_idx][:, 0]
    jaccard = get_jaccard_from_df(val_df, start_idx.astype('int'), end_idx.astype('int'), 'roberta', None)
    print(f'\nMax joint probability jaccard: {jaccard}\n')


def max_start_end_ensemble():
    max_l = Config.Train.max_len
    val_df = pd.read_csv(Config.validation_path)
    max_start_idx_prob = np.zeros((val_df.shape[0], 5))
    max_end_idx_prob = np.zeros((val_df.shape[0], 5))
    start_idx = np.zeros((val_df.shape[0],), dtype='int')
    end_idx = np.zeros((val_df.shape[0],), dtype='int')
    s_idxs = np.zeros((val_df.shape[0], 5), dtype='int')
    e_idxs = np.zeros((val_df.shape[0], 5), dtype='int')
    models_paths = list(Path(Config.Train.checkpoint_dir / Config.model_type).iterdir())
    for i, path in enumerate(models_paths):
        tf.keras.backend.clear_session()

        _generator = RobertaDataGenerator(val_df, augment=False)
        dataset = tf.data.Dataset.from_generator(_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
        dataset = dataset.padded_batch(Config.Train.batch_size,
                                       padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                      {'sts': [max_l], 'ets': [max_l]}),
                                       padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                       {'sts': 0, 'ets': 0}))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        model = get_roberta()
        model.load_weights(str(path))

        s_idx, e_idx = model.predict(dataset, verbose=1)
        max_start_idx_prob[:, i] = np.max(s_idx, axis=-1)
        max_end_idx_prob[:, i] = np.max(e_idx, axis=-1)
        s_idxs[:, i] = np.argmax(s_idx, axis=-1)
        e_idxs[:, i] = np.argmax(e_idx, axis=-1)

    for i in range(val_df.shape[0]):
        cross = max_start_idx_prob[i][:, np.newaxis] * max_end_idx_prob[i][:, np.newaxis].T
        s, e = np.unravel_index(np.argmax(cross), cross.shape)
        start_idx[i] = s_idxs[i][s]
        end_idx[i] = e_idxs[i][e]
    jaccard = get_jaccard_from_df(val_df, start_idx.astype('int'), end_idx.astype('int'), 'roberta', None)
    print(f'\nMax joint probability jaccard: {jaccard}\n')


if __name__ == '__main__':
    base_ensemble()
    max_joint_proba_ensemble()
    max_start_end_ensemble()
