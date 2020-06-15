import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import random
import warnings
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras

from config import Config
from data_utils import RobertaDataGenerator, RobertaClassificationDataGenerator, BertDataGenerator
from data_utils import XLNetDataGenerator, AlbertDataGenerator
from models import get_roberta, get_classification_roberta, get_bert, get_xlnet, get_electra, get_albert
from utils import get_jaccard_from_df, get_steps
from custom_callbacks.warmup_cosine_decay import WarmUpCosineDecayScheduler


def train_roberta(train_df, val_df, fold_i, augment=False):
    max_l = Config.Train.max_len
    _train_generator = RobertaDataGenerator(train_df, augment=augment)
    train_dataset = tf.data.Dataset.from_generator(_train_generator.generate,
                                                   output_types=(
                                                       {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                       {'sts': tf.int32, 'ets': tf.int32}))
    train_dataset = train_dataset.padded_batch(Config.Train.batch_size,
                                               padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                              {'sts': [None], 'ets': [None]}),
                                               padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                               {'sts': 0, 'ets': 0}))
    train_dataset = train_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    _val_generator = RobertaDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.Train.batch_size,
                                           padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                          {'sts': [None], 'ets': [None]}),
                                           padding_values=({'ids': 1, 'att': 0, 'tti': 0},
                                                           {'sts': 0, 'ets': 0}))
    val_dataset = val_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    model = get_roberta()
    if fold_i == 0:
        model.summary()
    model_name = f'weights_v{Config.version}_f{fold_i + 1}.h5'

    train_spe = get_steps(train_df)
    val_spe = get_steps(val_df)

    tf_log_dir = Config.Train.tf_log_dir / f'roberta/fold{fold_i}'
    tf_log_dir.mkdir(parents=True, exist_ok=True)
    cbs = [
        WarmUpCosineDecayScheduler(6e-5, 1200, warmup_steps=300, hold_base_rate_steps=200, verbose=0),
        keras.callbacks.ModelCheckpoint(
            str(Config.Train.checkpoint_dir / Config.model_type / model_name),
            verbose=1, save_best_only=True, save_weights_only=True),
        keras.callbacks.TensorBoard(log_dir=tf_log_dir, histogram_freq=2, update_freq=train_spe * 2, profile_batch=0)
    ]
    model.fit(train_dataset, epochs=2, verbose=1,
              validation_data=val_dataset, callbacks=cbs,
              steps_per_epoch=train_spe, validation_steps=val_spe)

    print(f'Loading checkpoint {model_name}...')
    model.load_weights(str(Config.Train.checkpoint_dir / Config.model_type / model_name))

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
    s_idx, e_idx = model.predict(val_dataset, verbose=1)
    s_idx = np.argmax(s_idx, axis=-1)
    e_idx = np.argmax(e_idx, axis=-1)
    jaccard_score = get_jaccard_from_df(val_df, s_idx, e_idx, 'roberta', 'roberta.csv')
    print(f'\n>>> Fold {fold_i + 1}: jaccard_score for roberta: {jaccard_score}\n')
    return jaccard_score


def train_bert(train_df, val_df, augment=False):
    max_l = Config.Train.max_len
    _train_generator = BertDataGenerator(train_df, augment=augment)
    train_dataset = tf.data.Dataset.from_generator(_train_generator.generate,
                                                   output_types=(
                                                       {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                       {'sts': tf.int32, 'ets': tf.int32}))
    train_dataset = train_dataset.padded_batch(Config.Train.batch_size,
                                               padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                              {'sts': [None], 'ets': [None]}))
    train_dataset = train_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    _val_generator = BertDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.Train.batch_size,
                                           padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                          {'sts': [None], 'ets': [None]}))
    val_dataset = val_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    model = get_bert()
    model.summary()
    model_name = f'weights_v{Config.version}.h5'

    train_spe = get_steps(train_df)
    oof_spe = get_steps(val_df)

    cbs = [
        WarmUpCosineDecayScheduler(6e-5, 1200, warmup_steps=300, hold_base_rate_steps=200, verbose=0),
        keras.callbacks.ModelCheckpoint(
            str(Config.Train.checkpoint_dir / Config.model_type / model_name),
            verbose=1, save_best_only=True, save_weights_only=True)
    ]
    model.fit(train_dataset, epochs=2, verbose=1,
              validation_data=val_dataset, callbacks=cbs,
              steps_per_epoch=train_spe, validation_steps=oof_spe)

    _val_generator = BertDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.Train.batch_size,
                                           padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                          {'sts': [max_l], 'ets': [max_l]}))
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    s_idx, e_idx = model.predict(val_dataset, verbose=1)
    s_idx = np.argmax(s_idx, axis=-1)
    e_idx = np.argmax(e_idx, axis=-1)
    jaccard_score = get_jaccard_from_df(val_df, s_idx, e_idx, 'bert', 'bert.csv')
    print(f'\n>>> jaccard_score for bert: {jaccard_score}\n')


def train_xlnet(train_df, val_df, augment=False):
    max_l = Config.Train.max_len
    _train_generator = XLNetDataGenerator(train_df, augment=augment)
    train_dataset = tf.data.Dataset.from_generator(_train_generator.generate,
                                                   output_types=(
                                                       {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                       {'sts': tf.int32, 'ets': tf.int32}))
    train_dataset = train_dataset.padded_batch(Config.XLNet.batch_size,
                                               padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                              {'sts': [None], 'ets': [None]}))
    train_dataset = train_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    _val_generator = XLNetDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.XLNet.batch_size,
                                           padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                          {'sts': [None], 'ets': [None]}))
    val_dataset = val_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    model = get_xlnet()
    model.summary()
    model_name = f'weights_v{Config.version}.h5'

    train_spe = get_steps(train_df)
    oof_spe = get_steps(val_df)

    cbs = [
        WarmUpCosineDecayScheduler(6e-5, 1200, warmup_steps=300, hold_base_rate_steps=200, verbose=0),
        keras.callbacks.ModelCheckpoint(
            str(Config.Train.checkpoint_dir / Config.model_type / model_name),
            verbose=1, save_best_only=True, save_weights_only=True)
    ]
    model.fit(train_dataset, epochs=2, verbose=1,
              validation_data=val_dataset, callbacks=cbs,
              steps_per_epoch=train_spe, validation_steps=oof_spe)

    _val_generator = XLNetDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.XLNet.batch_size,
                                           padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                          {'sts': [max_l], 'ets': [max_l]}))
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    s_idx, e_idx = model.predict(val_dataset, verbose=1)
    s_idx = np.argmax(s_idx, axis=-1)
    e_idx = np.argmax(e_idx, axis=-1)
    jaccard_score = get_jaccard_from_df(val_df, s_idx, e_idx, 'xlnet', 'xlnet.csv')
    print(f'\n>>> jaccard_score for xlnet: {jaccard_score}\n')


def train_electra(train_df, val_df, augment=False):
    max_l = Config.Train.max_len
    _train_generator = BertDataGenerator(train_df, augment=augment)
    train_dataset = tf.data.Dataset.from_generator(_train_generator.generate,
                                                   output_types=(
                                                       {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                       {'sts': tf.int32, 'ets': tf.int32}))
    train_dataset = train_dataset.padded_batch(Config.Train.batch_size,
                                               padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                              {'sts': [None], 'ets': [None]}))
    train_dataset = train_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    _val_generator = BertDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.Train.batch_size,
                                           padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                          {'sts': [None], 'ets': [None]}))
    val_dataset = val_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    model = get_electra()
    model.summary()
    model_name = f'weights_v{Config.version}.h5'

    train_spe = get_steps(train_df)
    oof_spe = get_steps(val_df)

    cbs = [
        WarmUpCosineDecayScheduler(6e-5, 1200, warmup_steps=300, hold_base_rate_steps=200, verbose=0),
        keras.callbacks.ModelCheckpoint(
            str(Config.Train.checkpoint_dir / Config.model_type / model_name),
            verbose=1, save_best_only=True, save_weights_only=True)
    ]
    model.fit(train_dataset, epochs=2, verbose=1,
              validation_data=val_dataset, callbacks=cbs,
              steps_per_epoch=train_spe, validation_steps=oof_spe)

    _val_generator = BertDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.Train.batch_size,
                                           padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                          {'sts': [max_l], 'ets': [max_l]}))
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    s_idx, e_idx = model.predict(val_dataset, verbose=1)
    s_idx = np.argmax(s_idx, axis=-1)
    e_idx = np.argmax(e_idx, axis=-1)
    jaccard_score = get_jaccard_from_df(val_df, s_idx, e_idx, 'bert', 'electra.csv')
    print(f'\n>>> jaccard_score for electra: {jaccard_score}\n')


def train_albert(train_df, val_df, fold_i, augment=False):
    max_l = Config.Albert.max_len
    _train_generator = AlbertDataGenerator(train_df, augment=augment)
    train_dataset = tf.data.Dataset.from_generator(_train_generator.generate,
                                                   output_types=(
                                                       {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                       {'sts': tf.int32, 'ets': tf.int32}))
    train_dataset = train_dataset.padded_batch(Config.Train.batch_size,
                                               padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                              {'sts': [None], 'ets': [None]}))
    train_dataset = train_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    _val_generator = AlbertDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.Train.batch_size,
                                           padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
                                                          {'sts': [None], 'ets': [None]}))
    val_dataset = val_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    model = get_albert()
    if fold_i == 0:
        model.summary()
    model_name = f'weights_v{Config.version}_f{fold_i + 1}.h5'

    train_spe = get_steps(train_df)
    val_spe = get_steps(val_df)

    cbs = [
        WarmUpCosineDecayScheduler(6e-5, 1200, warmup_steps=300, hold_base_rate_steps=200, verbose=0),
        keras.callbacks.ModelCheckpoint(
            str(Config.Train.checkpoint_dir / Config.model_type / model_name),
            verbose=1, save_best_only=True, save_weights_only=True)
    ]
    model.fit(train_dataset, epochs=2, verbose=1,
              validation_data=val_dataset, callbacks=cbs,
              steps_per_epoch=train_spe, validation_steps=val_spe)

    print(f'Loading checkpoint {model_name}...')
    model.load_weights(str(Config.Train.checkpoint_dir / Config.model_type / model_name))

    _val_generator = AlbertDataGenerator(val_df, augment=False)
    val_dataset = tf.data.Dataset.from_generator(_val_generator.generate,
                                                 output_types=(
                                                     {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
                                                     {'sts': tf.int32, 'ets': tf.int32}))
    val_dataset = val_dataset.padded_batch(Config.Train.batch_size,
                                           padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]},
                                                          {'sts': [max_l], 'ets': [max_l]}))
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    s_idx, e_idx = model.predict(val_dataset, verbose=1)
    s_idx = np.argmax(s_idx, axis=-1)
    e_idx = np.argmax(e_idx, axis=-1)
    jaccard_score = get_jaccard_from_df(val_df, s_idx, e_idx, 'albert', 'albert.csv')
    print(f'\n>>> Fold {fold_i + 1}: jaccard_score for albert: {jaccard_score}\n')
    return jaccard_score


def train(model_type):
    data_df = pd.read_csv(Config.data_path).dropna()
    data_df = data_df.sample(frac=1, random_state=Config.seed).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=Config.Train.n_folds)
    splits = skf.split(data_df.text.values, data_df.sentiment.values)
    jaccards = []
    for i, (train_idx, val_idx) in enumerate(splits):
        train_df = data_df.iloc[train_idx]
        val_df = data_df.iloc[val_idx]
        train_df = train_df.reindex(train_df.text.str.len().sort_values().index).reset_index(drop=True)
        val_df = val_df.reindex(val_df.text.str.len().sort_values().index).reset_index(drop=True)
        if model_type == 'roberta':
            jaccard = train_roberta(train_df, val_df, i, augment=Config.Train.augment)
            jaccards.append(jaccard)
        elif model_type == 'bert':
            train_bert(train_df, val_df, augment=Config.Train.augment)
        elif model_type == 'xlnet':
            train_xlnet(train_df, val_df, augment=Config.Train.augment)
        elif model_type == 'electra':
            train_electra(train_df, val_df, augment=Config.Train.augment)
        elif model_type == 'albert':
            jaccard = train_albert(train_df, val_df, i, augment=Config.Train.augment)
            jaccards.append(jaccard)
    print(f'>>> Mean jaccard for {model_type}: {np.mean(jaccards)}')


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
    start_time = time()
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    tf.random.set_seed(Config.seed)
    warnings.filterwarnings('ignore')

    shutil.rmtree(str(Config.Train.tf_log_dir / Config.model_type))

    tf.config.optimizer.set_jit(Config.Train.use_xla)
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": Config.Train.use_amp})

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
        train('roberta')
    elif Config.model_type == 'classification_roberta':
        train_roberta_classifier()
    elif Config.model_type == 'bert':
        train('bert')
    elif Config.model_type == 'xlnet':
        train('xlnet')
    elif Config.model_type == 'electra':
        train('electra')
    elif Config.model_type == 'albert':
        train('albert')

    end_time = time() - start_time
    print(f'Training took {end_time} seconds ({end_time / 60} minutes).')
