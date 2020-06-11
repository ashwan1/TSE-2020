import random

import numpy as np
import pandas as pd
from tensorflow import keras

from augmentation import synonym_replacement, random_char_repeat, random_char_deletion
from config import Config
from utils import get_tokenizer, get_ft_embeddings


class RobertaDataGenerator:
    def __init__(self, data: pd.DataFrame, augment: bool = False):
        self._augment = augment
        self._tokenizer = get_tokenizer('roberta')
        self._sentiment_ids = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
        self._data_df = data
        self.exception_count = 0
        self.exceptions = []
        self.exception_mask = []

    def generate(self):
        for row in self._data_df.itertuples(index=False, name='tweet'):
            text: str = row.text.lower()
            selected_text: str = row.selected_text.lower()
            if self._augment and random.random() > 0.5:
                text_list = text.split()
                n = random.choice([0, 1, 2, 3])
                if n == 0:
                    text_list, change_logs = synonym_replacement(text_list, 2)
                    text = ' '.join(text_list)
                    for k, v in change_logs.items():
                        selected_text = selected_text.replace(k, v)
                elif n == 1:
                    text_list, change_logs = random_char_repeat(text_list)
                    text = ' '.join(text_list)
                    for k, v in change_logs.items():
                        selected_text = selected_text.replace(k, v)
                elif n == 2:
                    text_list, change_logs = random_char_deletion(text_list)
                    text = ' '.join(text_list)
                    for k, v in change_logs.items():
                        selected_text = selected_text.replace(k, v)
                else:
                    text = ' '.join(text_list)
            # find overlap
            text = ' ' + ' '.join(text.split())
            selected_text = ' '.join(selected_text.split())
            idx_selected_text = text.find(selected_text)
            chars = np.zeros((len(text)))
            chars[idx_selected_text:idx_selected_text + len(selected_text)] = 1
            if text[idx_selected_text - 1] == ' ':
                chars[idx_selected_text - 1] = 1
            encoded_text = self._tokenizer.encode(text)
            # Id offsets
            offsets = []
            idx = 0
            for t in encoded_text.ids:
                w = self._tokenizer.decode([t])
                len_w = len(w)
                offsets.append((idx, idx + len_w))
                idx += len_w
            # Start end tokens
            tokens = []
            for i, (a, b) in enumerate(offsets):
                sm = np.sum(chars[a:b])
                if sm > 0:
                    tokens.append(i)
            sentiment_token = self._sentiment_ids[row.sentiment]
            # below [2] is token id for </s> token
            input_ids = [0] + encoded_text.ids + [2, 2] + [sentiment_token] + [2]
            len_encoded_ids = len(encoded_text.ids)
            attention_mask = [1] * (len_encoded_ids + 5)
            token_type_ids = [0] * (len_encoded_ids + 5)
            if len(tokens) > 0:
                self.exception_mask.append(True)
                start_tokens = np.zeros((len_encoded_ids + 5), dtype='int')
                start_tokens[tokens[0] + 1] = 1
                end_tokens = np.zeros((len_encoded_ids + 5), dtype='int')
                end_tokens[tokens[-1] + 1] = 1
                start_tokens = start_tokens.tolist()
                end_tokens = end_tokens.tolist()
            else:
                self.exception_count += 1
                self.exceptions = {'text': text, 'selected_text': selected_text, 'sentiment': row.sentiment}
                self.exception_mask.append(False)
                continue
            yield ({'ids': input_ids, 'att': attention_mask, 'tti': token_type_ids},
                   {'sts': start_tokens, 'ets': end_tokens})


class RobertaTestDataGenerator:
    def __init__(self, data: pd.DataFrame, augment: bool = False):
        self._augment = augment
        self._tokenizer = get_tokenizer('roberta')
        self._sentiment_ids = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
        self._data_df = data
        self._data_df.fillna('')

    def generate(self):
        for row in self._data_df.itertuples(index=False, name='tweet'):
            text: str = row.text.lower()
            if self._augment and random.random() > 0.5:
                text_list = text.split()
                n = random.choice([0, 1, 2, 3])
                if n == 0:
                    text_list, change_logs = synonym_replacement(text_list, 2)
                    text = ' '.join(text_list)
                elif n == 1:
                    text_list, change_logs = random_char_repeat(text_list)
                    text = ' '.join(text_list)
                elif n == 2:
                    text_list, change_logs = random_char_deletion(text_list)
                    text = ' '.join(text_list)
                else:
                    text = ' '.join(text_list)
            text = ' ' + ' '.join(text.split())
            encoded_text = self._tokenizer.encode(text)
            sentiment_token = self._sentiment_ids[row.sentiment]
            # below [2] is token id for </s> token
            input_ids = [0] + encoded_text.ids + [2, 2] + [sentiment_token] + [2]
            len_encoded_ids = len(encoded_text.ids)
            attention_mask = [1] * (len_encoded_ids + 5)
            token_type_ids = [0] * (len_encoded_ids + 5)
            yield {'ids': input_ids, 'att': attention_mask, 'tti': token_type_ids}


class RobertaData:
    def __init__(self, df: pd.DataFrame):
        self._data_df = df
        self._tokenizer = get_tokenizer('roberta')
        self._sentiment_ids = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
        n_data = self._data_df.shape[0]
        self._input_ids = np.ones((n_data, Config.Train.max_len), dtype='int32')
        self._attention_mask = np.zeros((n_data, Config.Train.max_len), dtype='int32')
        self._token_type_ids = np.zeros((n_data, Config.Train.max_len), dtype='int32')
        self._start_tokens = np.zeros((n_data, Config.Train.max_len), dtype='int32')
        self._end_tokens = np.zeros((n_data, Config.Train.max_len), dtype='int32')

    def get_data(self):
        for i, row in enumerate(self._data_df.itertuples(index=False, name='tweet')):
            text: str = row.text
            selected_text: str = row.selected_text
            # find overlap
            text = ' ' + ' '.join(text.split())
            selected_text = ' '.join(selected_text.split())
            idx_selected_text = text.find(selected_text)
            chars = np.zeros((len(text)))
            chars[idx_selected_text:idx_selected_text + len(selected_text)] = 1
            if text[idx_selected_text - 1] == ' ':
                chars[idx_selected_text - 1] = 1
            encoded_text = self._tokenizer.encode(text)
            # Id offsets
            offsets = []
            idx = 0
            for t in encoded_text.ids:
                w = self._tokenizer.decode([t])
                len_w = len(w)
                offsets.append((idx, idx + len_w))
                idx += len_w
            # Start end tokens
            tokens = []
            for j, (a, b) in enumerate(offsets):
                sm = np.sum(chars[a:b])
                if sm > 0:
                    tokens.append(j)
            sentiment_token = self._sentiment_ids[row.sentiment]
            # below [2] is token id for </s> token
            len_encoded_ids = len(encoded_text.ids)
            self._input_ids[i, :len_encoded_ids + 5] = [0] + encoded_text.ids + [2, 2] + [sentiment_token] + [2]
            self._attention_mask[i, :len_encoded_ids + 5] = 1
            if len(tokens) > 0:
                self._start_tokens[i, tokens[0] + 1] = 1
                self._end_tokens[i, tokens[-1] + 1] = 1
        return [self._input_ids, self._attention_mask, self._token_type_ids], [self._start_tokens, self._end_tokens]


class WordLevelDataGenerator:
    def __init__(self, data: pd.DataFrame, augment: bool = False):
        self.exceptions = []
        self.mask = []
        self._augment = augment
        self._data_df = data
        self._embeddings = get_ft_embeddings()
        self.count = 0

    def generate(self):
        for row in self._data_df.itertuples(index=False, name='tweet'):
            text: str = row.text.lower()
            selected_text: str = row.selected_text.lower()
            if self._augment and random.random() > 0.5:
                text_list = text.split()
                n = random.choice([0, 1, 2, 3])
                if n == 0:
                    text_list, change_logs = synonym_replacement(text_list, 2)
                    text = ' '.join(text_list)
                    for k, v in change_logs.items():
                        selected_text = selected_text.replace(k, v)
                elif n == 1:
                    text_list, change_logs = random_char_repeat(text_list)
                    text = ' '.join(text_list)
                    for k, v in change_logs.items():
                        selected_text = selected_text.replace(k, v)
                elif n == 2:
                    text_list, change_logs = random_char_deletion(text_list)
                    text = ' '.join(text_list)
                    for k, v in change_logs.items():
                        selected_text = selected_text.replace(k, v)
                else:
                    text = ' '.join(text_list)
            # find overlap
            text = ' '.join(text.split())
            selected_text = ' '.join(selected_text.split())
            idx_selected_text = text.find(selected_text)
            chars = np.zeros((len(text)))
            chars[idx_selected_text:idx_selected_text + len(selected_text)] = 1

            # Id offsets
            offsets = []
            idx = 0
            for w in text.split():
                len_w = len(w)
                offsets.append((idx, idx + len_w))
                idx += len_w + 1

            # Start end tokens
            tokens = []
            for i, (a, b) in enumerate(offsets):
                sm = np.sum(chars[a:b])
                if sm > 0:
                    tokens.append(i)

            if len(tokens) > 0:
                self.mask.append(True)
                start_tokens = np.zeros(len(text.split()), dtype='int')
                start_tokens[tokens[0]] = 1
                end_tokens = np.zeros(len(text.split()), dtype='int')
                end_tokens[tokens[-1]] = 1
                start_tokens = start_tokens.tolist()
                end_tokens = end_tokens.tolist()
                text = f'{text} <senti> {row.sentiment}'
                inputs = np.zeros((len(text.split()), Config.ft_embeddings_size))
                for i, word in enumerate(text.split()):
                    try:
                        inputs[i] = self._embeddings.wv[word]
                    except KeyError:
                        inputs[i] = self._embeddings.wv['<unk>']
            else:
                self.count += 1
                self.exceptions.append({'text': text, 'selected_text': selected_text, 'sentiment': row.sentiment})
                self.mask.append(False)
                continue
            yield {'inputs': inputs.tolist()}, {'sts': start_tokens, 'ets': end_tokens}


class RobertaClassificationDataGenerator:
    def __init__(self, data: pd.DataFrame, augment: bool = False):
        self._augment = augment
        self._tokenizer = get_tokenizer('roberta')
        self._sentiment_ids = {'positive': 1, 'negative': 2, 'neutral': 0}
        self._data_df = data

    def generate(self):
        for row in self._data_df.itertuples(index=False, name='tweet'):
            text: str = row.text.lower()
            if self._augment and random.random() > 0.5:
                text_list = text.split()
                n = random.choice([0, 1, 2, 3])
                if n == 0:
                    text_list, change_logs = synonym_replacement(text_list, 2)
                    text = ' '.join(text_list)
                elif n == 1:
                    text_list, change_logs = random_char_repeat(text_list)
                    text = ' '.join(text_list)
                elif n == 2:
                    text_list, change_logs = random_char_deletion(text_list)
                    text = ' '.join(text_list)
                else:
                    text = ' '.join(text_list)
            text = ' ' + ' '.join(text.split())
            encoded_text = self._tokenizer.encode(text)
            sentiment_id = self._sentiment_ids[row.sentiment]
            # below [2] is token id for </s> token
            input_ids = [0] + encoded_text.ids + [2]
            len_encoded_ids = len(encoded_text.ids)
            attention_mask = [1] * (len_encoded_ids + 2)
            token_type_ids = [0] * (len_encoded_ids + 2)
            yield ({'ids': input_ids, 'att': attention_mask, 'tti': token_type_ids},
                   {'sentiment': keras.utils.to_categorical(sentiment_id, num_classes=3)})
