import random

import numpy as np
import pandas as pd
import tensorflow as tf

from config import Config
from data_utils import RobertaTestDataGenerator
from models.roberta import get_roberta
from utils import get_tokenizer


def predict_test():
    print('\n>> Predicting on test')
    max_l = Config.Train.max_len
    test_df = pd.read_csv(Config.test_path)
    _test_generator = RobertaTestDataGenerator(test_df)
    test_dataset = tf.data.Dataset.from_generator(_test_generator.generate,
                                                  output_types=(
                                                      {'ids': tf.int32, 'att': tf.int32, 'tti': tf.int32}))
    test_dataset = test_dataset.padded_batch(Config.Train.batch_size,
                                             padded_shapes=({'ids': [max_l], 'att': [max_l], 'tti': [max_l]}),
                                             padding_values=({'ids': 1, 'att': 0, 'tti': 0}))
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model_dir = Config.Train.checkpoint_dir / Config.model_type
    start_idx = 0
    end_idx = 0
    model_count = len(list(model_dir.iterdir()))
    for i in range(model_count):
        model_path = model_dir / f'weights_{Config.version}_{i}.h5'
        model = get_roberta()
        model.load_weights(str(model_path))
        start_idx, end_idx = model.predict(test_dataset, verbose=1)
        start_idx += start_idx
        end_idx += end_idx
    start_idx /= model_count
    end_idx /= model_count
    start_idx = np.argmax(start_idx, axis=-1)
    end_idx = np.argmax(end_idx, axis=-1)
    end_idx = np.where(start_idx > end_idx, start_idx, end_idx)
    tokenizer = get_tokenizer('roberta')
    selected_texts = []
    for i, row in enumerate(test_df.itertuples(index=False, name='tweet')):
        a = start_idx[i]
        b = end_idx[i]
        text = ' ' + ' '.join(row.text.split())
        encoded_text = tokenizer.encode(text)
        selected_text = tokenizer.decode(encoded_text.ids[a - 1:b])
        selected_texts.append(selected_text)
    test_df['selected_text'] = selected_texts
    test_df.to_csv('test_predictions.csv', index=False)
    test_df[['textID', 'selected_text']].to_csv('submission.csv', index=False)


if __name__ == '__main__':
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    tf.random.set_seed(Config.seed)
    predict_test()
