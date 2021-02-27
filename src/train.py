import os
from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.test_tube import TestTubeLogger

from data import SentimentLoader
from pipeline import Pipeline
# from tuner import args
from utils import save_args


def main(args):
    tt_logger = TestTubeLogger(save_dir=args.log_path, name="",
                               description=args.description, debug=False,
                               create_git_tag=args.git_tag)
    tt_logger.experiment

    log_dir = Path(tt_logger.save_dir) / f"version_{tt_logger.version}"

    checkpoint_dir = log_dir / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    chkpt_callback = ModelCheckpoint(checkpoint_dir,
                                     monitor='Loss/val_loss',
                                     save_last=True,
                                     mode='min',
                                     save_top_k=10,
                                     period=5
                                     )

    data_loader = SentimentLoader.from_argparse_args(args)

    model_pipeline = Pipeline.from_argparse_args(args)

    save_args(args, log_dir)

    trainer = Trainer.from_argparse_args(args, logger=tt_logger,
                                         checkpoint_callback=chkpt_callback,
                                         gradient_clip_val=1.0,
                                         #   early_stop_callback=False,
                                         weights_summary='full',
                                         gpus=1,
                                         profiler=True
                                         )

    trainer.fit(model_pipeline, data_loader)
    trainer.test(model_pipeline)


if __name__ == "__main__":

    if 'args' not in locals():
        parser = ArgumentParser()
        parser = Pipeline.add_model_specific_args(parser)
        parser = SentimentLoader.add_argparse_args(parser)
        parser = Pipeline.add_argparse_args(parser)
        parser = Trainer.add_argparse_args(parser)
        args = parser.parse_args()

    main(args)
