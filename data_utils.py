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


class BertDataGenerator:
    def __init__(self, data: pd.DataFrame, augment: bool = False):
        self._augment = augment
        self._tokenizer = get_tokenizer('bert')
        self._sentiment_ids = {'positive': 3893, 'negative': 4997, 'neutral': 8699}
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
            text = ' '.join(text.split())
            selected_text = ' '.join(selected_text.split())
            # find the intersection between text and selected text
            idx_start, idx_end = None, None
            for index in (i for i, c in enumerate(text) if c == selected_text[0]):
                if text[index:index + len(selected_text)] == selected_text:
                    idx_start = index
                    idx_end = index + len(selected_text)
                    break

            intersection = [0] * len(text)
            if idx_start is not None and idx_end is not None:
                self.exception_mask.append(True)
                for char_idx in range(idx_start, idx_end):
                    intersection[char_idx] = 1
            else:
                self.exception_count += 1
                self.exceptions = {'text': text, 'selected_text': selected_text, 'sentiment': row.sentiment}
                self.exception_mask.append(False)
                continue

            # tokenize with offsets
            enc = self._tokenizer.encode(text, add_special_tokens=False)
            input_ids_orig, offsets = enc.ids, enc.offsets

            # compute targets
            target_idx = []
            for i, (o1, o2) in enumerate(offsets):
                if sum(intersection[o1: o2]) > 0:
                    target_idx.append(i)

            start_tokens = target_idx[0]
            end_tokens = target_idx[-1]

            input_ids = [101] + [self._sentiment_ids[row.sentiment]] + [102] + input_ids_orig + [102]
            token_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
            attention_mask = [1] * (len(input_ids_orig) + 4)
            start_tokens += 3
            end_tokens += 3
            np_start_tokens = np.zeros((len(input_ids)), dtype='int')
            np_start_tokens[start_tokens] = 1
            np_end_tokens = np.zeros((len(input_ids)), dtype='int')
            np_end_tokens[end_tokens] = 1
            start_tokens = np_start_tokens.tolist()
            end_tokens = np_end_tokens.tolist()
            yield ({'ids': input_ids, 'att': attention_mask, 'tti': token_type_ids},
                   {'sts': start_tokens, 'ets': end_tokens})


class XLNetDataGenerator:
    def __init__(self, data: pd.DataFrame, augment: bool = False):
        self._augment = augment
        self._tokenizer = get_tokenizer('xlnet')
        self._sentiment_ids = {'positive': 1654, 'negative': 2981, 'neutral': 9201}
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
            text = ' '.join(text.split())
            selected_text = ' '.join(selected_text.split())
            # find the intersection between text and selected text
            idx_start = text.find(selected_text)

            chars = np.zeros((len(text)))
            chars[idx_start:idx_start + len(selected_text)] = 1

            # tokenize with offsets
            input_tokens = self._tokenizer.tokenize(text)
            offsets = []
            idx = 0
            for t in input_tokens:
                len_t = len(t)
                offsets.append((idx, idx + len_t))
                idx += len_t

            # compute targets
            target_idx = []
            for i, (o1, o2) in enumerate(offsets):
                if sum(chars[o1: o2]) > 0:
                    target_idx.append(i)

            start_tokens = target_idx[0]
            end_tokens = target_idx[-1]

            input_ids_orig = self._tokenizer.encode(text, add_special_tokens=False)
            input_ids = input_ids_orig + [4] + [self._sentiment_ids[row.sentiment]] + [4, 3]
            token_type_ids = [0] * (len(input_ids_orig) + 1) + [1, 1] + [2]
            attention_mask = [1] * (len(input_ids_orig) + 4)
            np_start_tokens = np.zeros((len(input_ids)), dtype='int')
            np_start_tokens[start_tokens] = 1
            np_end_tokens = np.zeros((len(input_ids)), dtype='int')
            np_end_tokens[end_tokens] = 1
            start_tokens = np_start_tokens.tolist()
            end_tokens = np_end_tokens.tolist()
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
