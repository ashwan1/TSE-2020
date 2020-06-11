import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras

from config import Config
from data_utils import RobertaDataGenerator, RobertaClassificationDataGenerator
from models.roberta import get_roberta, get_classification_roberta
from utils import get_jaccard_from_df, get_steps
from custom_callbacks.warmup_cosine_decay import WarmUpCosineDecayScheduler


def train_roberta():
    max_l = Config.Train.max_len
    train_df = pd.read_csv(Config.train_path).dropna()
    # val_df = pd.read_csv(Config.validation_path).dropna()
    # jaccards = []
    model_names = []
    skf = StratifiedKFold(Config.Train.n_folds, shuffle=True, random_state=Config.seed)
    skf_split = skf.split(train_df.text.values, train_df.sentiment.values)
    min_jaccard = 1.0
    for i, (train_idx, oof_idx) in enumerate(skf_split):
        print('=' * 50)
        print(f'# Fold {i + 1}')
        print('=' * 50)

        keras.backend.clear_session()

        _train_generator = RobertaDataGenerator(train_df.iloc[train_idx], augment=True)
        train_dataset = tf.data.Dataset.from_generator(_train_generator.generate,
                                                       output_types=(
                                                           {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                           {'sts': tf.int32, 'ets': tf.int32}))
        train_dataset = train_dataset.padded_batch(Config.Train.batch_size,
                                                   padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                                  {'sts': [None], 'ets': [None]}))
        train_dataset = train_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

        _oof_generator = RobertaDataGenerator(train_df.iloc[oof_idx], augment=False)
        oof_dataset = tf.data.Dataset.from_generator(_oof_generator.generate,
                                                     output_types=(
                                                         {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                         {'sts': tf.int32, 'ets': tf.int32}))
        oof_dataset = oof_dataset.padded_batch(Config.Train.batch_size,
                                               padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                              {'sts': [None], 'ets': [None]}))
        oof_dataset = oof_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

        model = get_roberta()
        if i == 0:
            model.summary()
        model_name = f'weights_v{Config.version}_f{i + 1}.h5'
        model_names.append(model_name)

        train_spe = get_steps(train_df.iloc[train_idx])
        oof_spe = get_steps(train_df.iloc[oof_idx])

        cbs = [
            WarmUpCosineDecayScheduler(6e-5, 1200, warmup_steps=300, hold_base_rate_steps=200, verbose=0),
            keras.callbacks.ModelCheckpoint(
                str(Config.Train.checkpoint_dir / Config.model_type / model_name),
                verbose=1, save_best_only=True, save_weights_only=True)
        ]
        model.fit(train_dataset, epochs=Config.Train.epochs, verbose=1,
                  validation_data=oof_dataset, callbacks=cbs,
                  steps_per_epoch=train_spe, validation_steps=oof_spe)

        _oof_generator = RobertaDataGenerator(train_df.iloc[oof_idx], augment=False)
        oof_dataset = tf.data.Dataset.from_generator(_oof_generator.generate,
                                                     output_types=(
                                                         {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                         {'sts': tf.int32, 'ets': tf.int32}))
        oof_dataset = oof_dataset.padded_batch(Config.Train.batch_size,
                                               padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                              {'sts': [max_l], 'ets': [max_l]}),
                                               padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                               {'sts': 0, 'ets': 0}))
        oof_dataset = oof_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        s_idx, e_idx = model.predict(oof_dataset, verbose=1)
        s_idx = np.argmax(s_idx, axis=-1)
        e_idx = np.argmax(e_idx, axis=-1)
        e_idx = np.where(s_idx > e_idx, s_idx, e_idx)
        jaccard_score = get_jaccard_from_df(train_df.iloc[oof_idx], s_idx, e_idx)
        if jaccard_score < min_jaccard:
            min_jaccard = jaccard_score
            train_df.iloc[train_idx].to_csv(Path('data/use_this_train.csv'), index=False)
            train_df.iloc[oof_idx].to_csv(Path('data/use_this_val.csv'), index=False)
        print(f'jaccard_score for {i+1}: {jaccard_score}')
        print(f'min_jaccard_score: {min_jaccard}')

    # assert Config.Train.n_folds == len(model_names)
    # start_idx = 0
    # end_idx = 0
    # for i, model_name in enumerate(model_names):
    #     print(f'\nLoading {model_name} weights...')
    #     keras.backend.clear_session()
    #     model = get_roberta()
    #     model.load_weights(str(Config.Train.checkpoint_dir / Config.model_type / model_name))
    #     print('Predicting on validation set')
    #     _val_generator = RobertaDataGenerator(val_df, augment=False)
    #     val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
    #                                                  output_types=(
    #                                                      {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
    #                                                      {'sts': tf.int32, 'ets': tf.int32}))
    #     val_dataset = val_dataset.padded_batch(Config.Train.batch_size,
    #                                            padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
    #                                                           {'sts': [max_l], 'ets': [max_l]}),
    #                                            padding_values=({'ids': 1, 'att': 0, 'tti': 0},
    #                                                            {'sts': 0, 'ets': 0}))
    #     val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #     s_idx, e_idx = model.predict(val_dataset, verbose=1)
    #     start_idx += s_idx
    #     end_idx += e_idx
    #     s_idx = np.argmax(s_idx, axis=-1)
    #     e_idx = np.argmax(e_idx, axis=-1)
    #     e_idx = np.where(s_idx > e_idx, s_idx, e_idx)
    #     jaccard_score = get_jaccard_from_df(val_df, s_idx, e_idx)
    #     jaccards.append(jaccard_score)
    #     print(f'\n>> Fold{i + 1}: Jaccard score on validation data: {jaccard_score}')
    #
    # print(f'\n>> Mean jaccard score for {Config.Train.n_folds} folds: {np.mean(jaccards)}\n')
    #
    # start_idx /= Config.Train.n_folds
    # end_idx /= Config.Train.n_folds
    # start_idx = np.argmax(start_idx, axis=-1)
    # end_idx = np.argmax(end_idx, axis=-1)
    #
    # start_gt_end_df = val_df[start_idx > end_idx]
    # start_gt_end_df['start_idx'] = start_idx[start_idx > end_idx]
    # start_gt_end_df['end_idx'] = end_idx[start_idx > end_idx]
    # start_gt_end_df.to_csv('start_gt_end.csv', index=False)
    #
    # end_idx = np.where(start_idx > end_idx, start_idx, end_idx)
    # jaccard_score = get_jaccard_from_df(val_df, start_idx, end_idx)
    # print(f'\n>> Ensemble: jaccard score on validation data: {jaccard_score}')
    # jaccard_score = get_jaccard_from_df(val_df, start_idx, end_idx, return_full_text_when_neutral=True)
    # print(f'\n>> Ensemble: jaccard score on validation data when full text is selected for '
    #       f'neutral sentiment: {jaccard_score}')
    # predict_test()


def train_roberta_classifier():
    max_l = Config.Train.max_len
    train_df = pd.read_csv(Config.train_path).dropna()
    val_df = pd.read_csv(Config.validation_path).dropna()
    _train_generator = RobertaClassificationDataGenerator(train_df, augment=True)
    train_dataset = tf.data.Dataset.from_generator(_train_generator.generate,
                                                   output_types=(
                                                       {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                       {'sentiment': tf.int32}))
    train_dataset = train_dataset.padded_batch(Config.Train.batch_size,
                                               padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                              {'sentiment': tf.compat.v1.Dimension(None)}),
                                               padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                               {'sentiment': 0}))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    _val_generator = RobertaClassificationDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sentiment': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.Train.batch_size,
                                           padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                          {'sentiment': tf.compat.v1.Dimension(None)}),
                                           padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                           {'sentiment': 0}))
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = get_classification_roberta()
    model.summary()

    cbs = [
        # keras.callbacks.ReduceLROnPlateau(patience=2, verbose=1, factor=0.3),
        keras.callbacks.EarlyStopping(patience=2, verbose=1, restore_best_weights=True),
        keras.callbacks.TensorBoard(log_dir=str(Config.Train.tf_log_dir / Config.model_type), histogram_freq=2,
                                    profile_batch=0, write_images=True),
        keras.callbacks.ModelCheckpoint(
            str(Config.Train.checkpoint_dir / Config.model_type / f'weights_v{Config.version}.h5'),
            verbose=1, save_best_only=True, save_weights_only=True)
    ]
    model.fit(train_dataset, epochs=50, verbose=1, validation_data=val_dataset, callbacks=cbs)

    print('\nLoading model weights...')
    model.load_weights(str(Config.Train.checkpoint_dir / Config.model_type / f'weights_v{Config.version}.h5'))

    print('\nEvaluating on validation set')
    _val_generator = RobertaClassificationDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sentiment': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.Train.batch_size,
                                           padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                          {'sentiment': tf.compat.v1.Dimension(None)}),
                                           padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                           {'sentiment': 0}))
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    model.evaluate(val_dataset, verbose=1)


if __name__ == '__main__':
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    tf.random.set_seed(Config.seed)
    warnings.filterwarnings('ignore')

    shutil.rmtree(str(Config.Train.tf_log_dir / Config.model_type))

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
    elif Config.model_type == 'classification_roberta':
        train_roberta_classifier()
