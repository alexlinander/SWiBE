import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.backbones.ncsnpp import NCSNpp
from src.data_module import SpecsDataModule
from src.sdes import OUVESDE
from src.model import ScoreModel


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups



if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--model_name", "-m", type=str, default="SWiBE")
          parser_.add_argument("--version", type=int, default=None)
          parser_.add_argument("--save_dir", type=str, required=True, help="directory to save model checkpoints and tensorboard logger")
          
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     parser = pl.Trainer.add_argparse_args(parser)
     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     OUVESDE.add_argparse_args(
          parser.add_argument_group("SDE", description=OUVESDE.__name__))
     NCSNpp.add_argparse_args(
          parser.add_argument_group("Backbone", description=NCSNpp.__name__))
     SpecsDataModule.add_argparse_args(
          parser.add_argument_group("DataModule", description=SpecsDataModule.__name__))
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)

     alpha = args.Alpha
     lambda_ = args.Lambda
     save_dir = args.save_dir
     model_name = args.model_name+"_alpha={}_lambda={}".format(alpha, lambda_)



     # Initialize logger, trainer, model, datamodule
     model = ScoreModel(sde=OUVESDE, backbone=NCSNpp, data_module_cls=SpecsDataModule,
          **{
               **vars(arg_groups['ScoreModel']),
               **vars(arg_groups['SDE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )

     model.plot(model_name, alpha, lambda_)
     logger = TensorBoardLogger(save_dir=f"{save_dir}/logs/{model_name}", name="tensorboard")


     # Set up callbacks for logger
     if args.version is None:
          callbacks = [ModelCheckpoint(dirpath=f"{save_dir}/logs/{model_name}", save_last=True, filename='{epoch}-last')]
          if args.num_eval_files:
               checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"{save_dir}/logs/{model_name}", 
                    save_top_k=1, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
               checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"{save_dir}/logs/{model_name}", 
                    save_top_k=1, monitor="lsd", mode="min", filename='{epoch}-{lsd:.2f}')
               callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr]

     else:
          callbacks = []
          if args.num_eval_files:
               checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"{save_dir}/logs/{model_name}/version_{str(args.version)}", 
                    save_top_k=1, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
               checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"{save_dir}/logs/{model_name}/version_{str(args.version)}", 
                    save_top_k=1, monitor="lsd", mode="min", filename='{epoch}-{lsd:.2f}')
               callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr]

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer.from_argparse_args(
          arg_groups['pl.Trainer'],
          strategy=DDPStrategy(find_unused_parameters=False), logger=logger,
          log_every_n_steps=10, num_sanity_val_steps=0,
          callbacks=callbacks,gpus=1, max_epochs=150
     )

     # Train model
     trainer.fit(model)
