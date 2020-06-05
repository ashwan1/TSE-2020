import random
from typing import List

import numpy as np
import tensorflow as tf
from kerastuner import HyperParameters
from kerastuner.tuners import Hyperband

from config import Config
from data_utils import RobertaDataGenerator, RobertaData
from models.roberta import get_tunable_roberta


def tune():
    # train_generator = RobertaDataGenerator(config.train_path)
    # train_dataset = tf.data.Dataset.from_generator(train_generator.generate,
    #                                                output_types=({'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
    #                                                              {'sts': tf.int32, 'ets': tf.int32}))
    # train_dataset = train_dataset.padded_batch(32,
    #                                            padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
    #                                                           {'sts': [None], 'ets': [None]}),
    #                                            padding_values=({'ids': 1, 'att': 0, 'tti': 0},
    #                                                            {'sts': 0, 'ets': 0}))
    # train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #
    # val_generator = RobertaDataGenerator(config.validation_path)
    # val_dataset = tf.data.Dataset.from_generator(val_generator.generate,
    #                                              output_types=({'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32},
    #                                                            {'sts': tf.int32, 'ets': tf.int32}))
    # val_dataset = val_dataset.padded_batch(32,
    #                                        padded_shapes=({'ids': [None], 'att': [None], 'tti': [None]},
    #                                                       {'sts': [None], 'ets': [None]}),
    #                                        padding_values=({'ids': 1, 'att': 0, 'tti': 0},
    #                                                        {'sts': 0, 'ets': 0}))
    # val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    train_dataset = RobertaData(Config.train_path, 'train').get_data()
    val_dataset = RobertaData(Config.validation_path, 'val').get_data()

    tuner = Hyperband(get_tunable_roberta,
                      objective='val_loss',
                      max_epochs=10,
                      factor=3,
                      hyperband_iterations=3,
                      seed=Config.seed,
                      directory='tuner_logs',
                      project_name='feat_roberta')

    tuner.search_space_summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)
    ]
    tuner.search(train_dataset[0], train_dataset[1], epochs=10,
                 verbose=1, callbacks=callbacks, batch_size=32,
                 validation_data=val_dataset)

    tuner.results_summary()
    best_hps: List[HyperParameters] = tuner.get_best_hyperparameters(num_trials=5)
    for hp in best_hps:
        print(f'{hp.values}\n')

    model = tuner.hypermodel.build(best_hps[0])
    tf.keras.utils.plot_model(model, to_file='best_hp_tuned_model.png', show_shapes=True,
                              show_layer_names=True,
                              expand_nested=True)
    model.summary()


if __name__ == '__main__':
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    tf.random.set_seed(Config.seed)

    tune()
