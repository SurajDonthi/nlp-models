from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import (
    f1_score, multiclass_auroc, precision, recall)
from transformers import (AdamW, BertModel, BertTokenizer, DistilBertModel,
                          DistilBertTokenizer, PreTrainedTokenizer,
                          SqueezeBertModel, SqueezeBertTokenizer,
                          get_linear_schedule_with_warmup)
from typing_extensions import Literal

from base import BaseModule
from models import ClassifierModel

LOSSES = {'bce': F.binary_cross_entropy,
          'bce_logits': F.binary_cross_entropy_with_logits,
          'cross_entropy': F.cross_entropy, 'nll_loss': F.nll_loss,
          'kl_div': F.kl_div, 'mse': F.mse_loss,
          'l1_loss': F.l1_loss}

BERT_BASE = {
    'bert': {'model': BertModel,
             'tokenizer': BertTokenizer,
             'pretrained_model_name': 'bert-base-cased',
             },
    'distil': {'model': DistilBertModel,
               'tokenizer': DistilBertTokenizer,
               'pretrained_model_name': 'distilbert-base-multilingual-cased'
               },
    'squeeze': {'model': SqueezeBertModel,
                'tokenizer': SqueezeBertTokenizer,
                'pretrained_model_name': 'squeezebert/squeezebert-uncased'
                }
}


class Pipeline(BaseModule):

    def __init__(self, bert_base: str = 'bert',
                 #  tokenizer: Optional[PreTrainedTokenizer] = None,
                 #  task: str = 'classification',
                 lr: float = 5e-5,
                 criterion: Literal[tuple(LOSSES.keys())] = 'cross_entropy',
                 freeze_bert: bool = False,
                 optim_args: dict = {'eps': 1e-8},
                 *args, **kwargs):
        super().__init__()

        self.criterion = LOSSES[criterion]
        self.optim_args = optim_args

        self.pretrained_model_name = \
            BERT_BASE[bert_base]['pretrained_model_name']

        self.bert_base = \
            BERT_BASE[bert_base]['model']\
            .from_pretrained(self.pretrained_model_name)
        self.tokenizer = \
            BERT_BASE[bert_base]['tokenizer']\
            .from_pretrained(self.pretrained_model_name)

        self.classifier = ClassifierModel()

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-des', '--description', required=False, type=str)
        parser.add_argument('-lp', '--log_path', type=str,
                            default='./logs')
        parser.add_argument('-gt', '--git_tag', required=False, const=False,
                            type=parsing.str_to_bool, nargs='?')
        parser.add_argument('--debug', required=False, const=False, nargs='?',
                            type=parsing.str_to_bool)
        return parser

    def configure_optimizers(self):
        params = []
        if self.hparams.freeze_bert:
            params += list(self.bert_base.parameters())
        params += list(self.classifier.parameters())

        optim = AdamW(params,
                      lr=self.hparams.lr, **self.optim_args)
        scheduler = get_linear_schedule_with_warmup(optim,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=self.trainer.max_epochs
                                                    )
        return [optim], [scheduler]

    def forward(self, batch):
        input_ids, attention_mask = batch

        embeddings = self.bert_base(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    # token_type_ids=token_type_ids
                                    )
        return self.classifier(embeddings)

    def shared_step(self, batch, return_preds=False):
        *batch, targets = batch
        out = self(batch)

        loss = self.criterion(out, targets)
        _, preds = torch.max(out, dim=1)
        acc = (preds == targets).float().mean()

        if return_preds:
            loss, acc, out, preds
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        logs = {
            'Loss/train_loss': loss,
            'Accuracy/train_acc': acc
        }
        self.log_dict(logs, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch:
            loss, acc = self.shared_step(batch)

            logs = {
                'Loss/val_loss': loss,
                'Accuracy/val_acc': acc
            }
            self.log_dict(logs, prog_bar=True, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        *_, targets = batch
        loss, acc, out, preds = self.shared_step(batch, return_preds=True)

        p = precision(out, targets)
        r = recall(out, targets)
        f1 = f1_score(preds, targets)
        auroc = multiclass_auroc(out, targets)

        logs = {
            'Loss/test_loss': loss,
            'Accuracy/test_acc': acc,
            'Precision': p,
            'Recall': r,
            'F1 score': f1,
            'Multiclass AUROC': auroc
        }
        self.log_dict(logs)
