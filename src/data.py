from pathlib import Path

import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer

from base import BaseDataModule


class SentimentDataset(Dataset):

    def __init__(self, file_path: str,
                 read_args: dict = dict(usecols=['review_body', 'sentiment']),
                 max_len=512, **kwargs
                 ):
        super().__init__()

        self.df = pd.read_csv(file_path, **read_args)

        self.max_len = max_len

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text, target = self.df.iloc[index]

        self.encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return \
            self.encoding.input_ids.squeeze(), \
            self.encoding.attention_mask.squeeze(), \
            th.tensor(target, dtype=th.long)
        # encoding['token_type_ids'].flatten(), \


class SentimentLoader(BaseDataModule):

    def __init__(self, data_path: str,
                 train_split_ratio: float = 0.7,
                 train_batchsize: int = 32,
                 val_batchsize: int = 32,
                 test_batchsize: int = 32,
                 num_workers: int = 4,
                 #  tokenizer=None,
                 data_args: dict = dict(max_len=512,
                                        read_args=dict(usecols=['review_body', 'sentiment'])
                                        ),
                 ):

        super().__init__()

        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise Exception(
                f"Path '{self.data_path.absolute().as_posix()}' does not exist!")

        self.train_split_ratio = train_split_ratio
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize
        self.val_batchsize = val_batchsize
        self.num_workers = num_workers
        self._data_args = data_args

    def prepare_data(self):
        self.train = SentimentDataset(self.data_path, **self._data_args)
        len_ = len(self.train)
        train_len = int(len_ * self.train_split_ratio)
        val_len = len_ - train_len
        print(f'Train length: {train_len}, Val length: {val_len}')

        self.train, self.val = random_split(self.train, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val:
            loader = DataLoader(self.val, batch_size=self.train_batchsize,
                                shuffle=True, num_workers=self.num_workers)
            return loader

    def test_dataloader(self):
        return self.val_dataloader()
