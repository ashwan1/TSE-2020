from gensim.models import FastText
from tokenizers import ByteLevelBPETokenizer

from config import Config
import pandas as pd
import numpy as np

from losses_n_metrics import jaccard

__roberta_tokenizer = ByteLevelBPETokenizer(vocab_file=str(Config.Roberta.vocab_file),
                                            merges_file=str(Config.Roberta.merges_file), add_prefix_space=True,
                                            lowercase=True)

__ft_embeddings = FastText.load(str(Config.ft_embeddings_path))


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
