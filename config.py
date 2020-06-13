from pathlib import Path

_root_dir = Path(__file__).parent


class Config:
    version = 0
    seed = 88888
    model_type = 'xlnet'

    train_path = _root_dir / 'data/use_this_train.csv'
    validation_path = _root_dir / 'data/use_this_val.csv'
    test_path = _root_dir / 'data/test.csv'

    ft_embeddings_path = _root_dir / 'embeddings/fasttext/twitter/twitter_ft.model'
    ft_embeddings_size = 200

    class Roberta:
        roberta_base_path = _root_dir / 'pretrained_models/distilroberta_base'
        vocab_file = roberta_base_path / 'vocab.json'
        merges_file = roberta_base_path / 'merges.txt'
        config = roberta_base_path / 'config.json'
        model = roberta_base_path / 'tf_model.h5'

    class Bert:
        epochs = 5
        bert_base_path = _root_dir / 'pretrained_models/bert_base'
        vocab_file = bert_base_path / 'vocab.txt'
        config = bert_base_path / 'config.json'
        model = bert_base_path / 'tf_model.h5'

    class XLNet:
        xlnet_base_path = _root_dir / 'pretrained_models/xlnet_base'
        vocab_file = xlnet_base_path / 'spiece.model'
        config = xlnet_base_path / 'config.json'
        model = xlnet_base_path / 'tf_model.h5'
        batch_size = 24

    class Train:
        augment = False
        use_xla = False
        use_amp = True
        n_folds = 5
        batch_size = 32
        max_len = 130
        label_smoothing = 0.1
        checkpoint_dir = _root_dir / 'checkpoints'
        tf_log_dir = _root_dir / 'tf_logs'


(Config.Train.checkpoint_dir / Config.model_type).mkdir(parents=True, exist_ok=True)
(Config.Train.tf_log_dir / Config.model_type).mkdir(parents=True, exist_ok=True)
