import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import FastText
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, SentencePieceBPETokenizer
from transformers import XLNetTokenizer, AlbertTokenizer

from config import Config
from losses_n_metrics import jaccard


def get_tokenizer(name: str):
    if name == 'roberta':
        return __roberta_tokenizer
    elif name == 'bert':
        return __bert_tokenizer
    elif name == 'xlnet':
        return __xlnet_tokenizer
    elif name == 'albert':
        return  __albert_tokenizer


def get_ft_embeddings():
    return __ft_embeddings


def get_steps(df):
    len_data = df.shape[0]
    if len_data % Config.Train.batch_size == 0:
        steps: int = len_data // Config.Train.batch_size
    else:
        steps: int = (len_data // Config.Train.batch_size) + 1
    return steps


def _get_selected_text_for_roberta(text, a, b, tokenizer):
    encoded_text = tokenizer.encode(text)
    pred_selected_text = tokenizer.decode(encoded_text.ids[a - 1:b])
    return pred_selected_text


def _get_selected_text_for_bert(text, a, b, tokenizer):
    a -= 3
    b -= 3
    encoded_text = tokenizer.encode(text, add_special_tokens=False)
    offset = encoded_text.offsets
    pred_selected_text = ''
    for i in range(a, b + 1):
        pred_selected_text += text[offset[i][0]:offset[i][1]]
        if (i + 1) < len(offset) and offset[i][1] < offset[i + 1][0]:
            pred_selected_text += " "
    if len(pred_selected_text) == 0:
        pred_selected_text = text
    return pred_selected_text


def _get_selected_text_for_xlnet(text, a, b, tokenizer):
    encoded_text = tokenizer.tokenize(text)
    pred_selected_text = tokenizer.convert_tokens_to_string(encoded_text[a: b + 1])
    return pred_selected_text


def get_jaccard_from_df(df: pd.DataFrame, start_idx: np.ndarray, end_idx: np.ndarray,
                        model_type: str, pred_file) -> float:
    assert start_idx.shape == (df.shape[0],), f'start_idx.shape={start_idx.shape}; df.shape={df.shape}'
    assert end_idx.shape == start_idx.shape, f'end_idx.shape={end_idx.shape}; start_idx.shape={start_idx.shape}'
    tokenizer = get_tokenizer(model_type)
    jaccards = []
    pred_selected_texts = []
    text_tokens = []
    for i, row in enumerate(df.itertuples(index=False, name='tweet')):
        a = start_idx[i]
        b = end_idx[i]
        if a > b:
            b = a  # Todo: Do something about this condition
        if model_type == 'roberta':
            text = ' ' + ' '.join(row.text.lower().split())
            pred_selected_text = _get_selected_text_for_roberta(text, a, b, tokenizer)
            tokens = tokenizer.encode(text).tokens
        elif model_type == 'bert':
            text = ' '.join(row.text.lower().split())
            pred_selected_text = _get_selected_text_for_bert(text, a, b, tokenizer)
            tokens = tokenizer.encode(text).tokens
        elif model_type == 'xlnet' or model_type == 'albert':
            text = ' '.join(row.text.lower().split())
            pred_selected_text = _get_selected_text_for_xlnet(text, a, b, tokenizer)
            tokens = tokenizer.tokenize(text)
        # if row.sentiment.lower() == 'neutral':
        #     pred_selected_text = row.selected_text
        jaccards.append(jaccard(row.selected_text, pred_selected_text))
        pred_selected_texts.append(pred_selected_text)
        text_tokens.append(tokens)
    df['jaccard'] = jaccards
    df['pred_selected_text'] = pred_selected_texts
    df['text_tokens'] = text_tokens
    if pred_file is not None:
        df.to_csv(Path(f'{Config.pred_dir}/{pred_file}'), index=False)
    return float(np.mean(jaccards))


__roberta_tokenizer = ByteLevelBPETokenizer(vocab_file=str(Config.Roberta.vocab_file),
                                            merges_file=str(Config.Roberta.merges_file), add_prefix_space=True,
                                            lowercase=True)

__bert_tokenizer = BertWordPieceTokenizer(vocab_file=str(Config.Bert.vocab_file))

__xlnet_tokenizer = XLNetTokenizer(vocab_file=str(Config.XLNet.vocab_file), do_lower_case=False)

__albert_tokenizer = AlbertTokenizer(vocab_file=str(Config.Albert.vocab_file), do_lower_case=True)

__ft_embeddings = FastText.load(str(Config.ft_embeddings_path))
