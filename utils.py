import string

from gensim.models import FastText
from sklearn.feature_extraction.text import CountVectorizer
from tokenizers import ByteLevelBPETokenizer

from config import Config
import pandas as pd
import numpy as np

from losses_n_metrics import jaccard


def get_tokenizer(name: str):
    if name == 'roberta':
        return __roberta_tokenizer


def get_ft_embeddings():
    return __ft_embeddings


def get_train_steps():
    df = pd.read_csv(Config.train_path)
    return _get_steps(df.shape[0])


def get_validation_steps():
    df = pd.read_csv(Config.validation_path)
    return _get_steps(df.shape[0])


def _get_steps(len_data: int) -> int:
    if len_data % Config.Train.batch_size == 0:
        steps: int = len_data // Config.Train.batch_size
    else:
        steps: int = (len_data // Config.Train.batch_size) + 1
    return steps


def get_jaccard_from_df(df: pd.DataFrame, start_idx: np.ndarray, end_idx: np.ndarray) -> float:
    assert start_idx.shape == (df.shape[0],), f'start_idx.shape={start_idx.shape}; df.shape={df.shape}'
    assert end_idx.shape == start_idx.shape, f'end_idx.shape={end_idx.shape}; start_idx.shape={start_idx.shape}'
    tokenizer = get_tokenizer('roberta')
    jaccards = []
    for i, row in enumerate(df.itertuples(index=False, name='tweet')):
        a = start_idx[i]
        b = end_idx[i]
        text = ' ' + ' '.join(row.text.split())
        encoded_text = tokenizer.encode(text)
        pred_selected_text = tokenizer.decode(encoded_text.ids[a - 1:b])
        # if row.sentiment.lower() == 'neutral':
        #     pred_selected_text = row.selected_text
        jaccards.append(jaccard(row.selected_text, pred_selected_text))
    return float(np.mean(jaccards))


def calculate_selected_text(df_row, tol=0):
    tweet = df_row.text.lower()
    sentiment = df_row.sentiment

    if sentiment == 'neutral':
        return tweet
    elif sentiment == 'positive':
        dict_to_use = __polarity_words['pos']  # Calculate word weights using the pos_words dictionary
    elif sentiment == 'negative':
        dict_to_use = __polarity_words['neg']  # Calculate word weights using the neg_words dictionary

    words = tweet.split()
    words_len = len(words)
    subsets = [words[i:j + 1] for i in range(words_len) for j in range(i, words_len)]

    score = 0
    selection_str = ''  # This will be our choice
    lst = sorted(subsets, key=len)  # Sort candidates by length

    for i in range(len(subsets)):

        new_sum = 0  # Sum for the current substring

        # Calculate the sum of weights for each word in the substring
        for p in range(len(lst[i])):
            # noinspection PyUnboundLocalVariable
            if lst[i][p].translate(str.maketrans('', '', string.punctuation)) in dict_to_use.keys():
                new_sum += dict_to_use[lst[i][p].translate(str.maketrans('', '', string.punctuation))]

        # If the sum is greater than the score, update our current selection
        if new_sum > score + tol:
            score = new_sum
            selection_str = lst[i]
            # tol = tol*5 # Increase the tolerance a bit each time we choose a selection

    # If we didn't find good substrings, return the whole text
    if len(selection_str) == 0:
        selection_str = words

    return ' '.join(selection_str)


def _gather_polarity_words():
    df = pd.read_csv(Config.train_path).dropna()[['text', 'sentiment']]
    df = df.append(pd.read_csv(Config.validation_path).dropna()[['text', 'sentiment']], ignore_index=True)
    df = df.append(pd.read_csv(Config.validation_path).dropna()[['text', 'sentiment']], ignore_index=True)

    pos_df = df[df['sentiment'] == 'positive']
    neutral_df = df[df['sentiment'] == 'neutral']
    neg_df = df[df['sentiment'] == 'negative']

    cv = CountVectorizer(max_df=0.95, min_df=2,
                         max_features=10000, lowercase=True,
                         stop_words='english')
    # noinspection PyUnusedLocal
    df_cv = cv.fit_transform(df['text'])
    x_pos = cv.transform(pos_df['text'])
    x_neutral = cv.transform(neutral_df['text'])
    x_neg = cv.transform(neg_df['text'])
    pos_count_df = pd.DataFrame(x_pos.toarray(), columns=cv.get_feature_names())
    neutral_count_df = pd.DataFrame(x_neutral.toarray(), columns=cv.get_feature_names())
    neg_count_df = pd.DataFrame(x_neg.toarray(), columns=cv.get_feature_names())

    pos_words = {}
    neutral_words = {}
    neg_words = {}

    for k in cv.get_feature_names():
        pos = pos_count_df[k].sum()
        neutral = neutral_count_df[k].sum()
        neg = neg_count_df[k].sum()

        pos_words[k] = pos / pos_df.shape[0]
        neutral_words[k] = neutral / neutral_df.shape[0]
        neg_words[k] = neg / neg_df.shape[0]

    neg_words_adj = {}
    pos_words_adj = {}
    neutral_words_adj = {}

    for key, value in neg_words.items():
        neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])

    for key, value in pos_words.items():
        pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])

    for key, value in neutral_words.items():
        neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])

    return {'neg': neg_words_adj, 'pos': pos_words_adj, 'neutral': neutral_words_adj}


__polarity_words = _gather_polarity_words()

__roberta_tokenizer = ByteLevelBPETokenizer(vocab_file=str(Config.Roberta.vocab_file),
                                            merges_file=str(Config.Roberta.merges_file), add_prefix_space=True,
                                            lowercase=True)

__ft_embeddings = FastText.load(str(Config.ft_embeddings_path))
