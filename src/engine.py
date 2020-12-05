from argparse import ArgumentParser
import pytorch_lightning as pl

from .model import Model

LOSSES = {'bce': F.binary_cross_entropy,
          'bce_logits': F.binary_cross_entropy_with_logits,
          'cross_entropy': F.cross_entropy, 'nll_loss': F.nll_loss,
          'kl_div': F.kl_div, 'mse': F.mse_loss,
          'l1_loss': F.l1_loss}


class Engine(Model, pl.LightningModule):

    def __init__(self, learning_rate=0.0001, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-des', '--description', required=False, type=str)
        parser.add_argument('-lp', '--log_path', type=str,
                            default='./lightning_logs')
        parser.add_argument('-lr', '--learning_rate',
                            type=float, default=0.0001)
        parser.add_argument('-c', '--criterion', type=str,
                            choices=LOSSES.keys(),
                            default='cross_entropy')
        parser.add_argument()
        return parser

    def configure_optimizers(self):
        optim = None
        scheduler = None
        return optim, scheduler

    def training_step(self, X):

        loss = None
        self.log(loss, progress_bar=True)

    def validation_step(self, X):

        loss = None
        self.log(loss, progress_bar=True)

    def test_step(self, X):

        loss = None
        self.log(loss, progress_bar=True)
