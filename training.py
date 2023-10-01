# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
# from pytorch_lightning.loggers import TensorBoardLogger

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule
from imuposer.utils import get_parser
from imuposer.datasets import GlobalModelDataset

from argparse import ArgumentParser

# set the random seed
seed_everything(42, workers=True)

# can also use ArgumentParser
parser = get_parser()
args = parser.parse_args()
combo_id = args.combo_id
fast_dev_run = args.fast_dev_run
_experiment = args.experiment

# %%
config = Config(experiment=f"{_experiment}_{combo_id}", model="GlobalModelIMUPoser",
                project_root_dir="../../", joints_set=amass_combos[combo_id], normalize="no_translation",
                r6d=True, loss_type="mse", use_joint_loss=True, device="0") 

# %%
# instantiate model and data
model = get_model(config)
datamodule = get_datamodule(config)
checkpoint_path = config.checkpoint_path 

# %%
# wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)
logger = TensorBoardLogger(save_dir=config.log_path, name=config.experiment)

early_stopping_callback = EarlyStopping(monitor="validation_step_loss", mode="min", verbose=False,
                                        min_delta=0.00001, patience=10)
checkpoint_callback = ModelCheckpoint(monitor="validation_step_loss", mode="min", verbose=False, 
                                      save_top_k=1, dirpath=checkpoint_path, save_weights_only=True, 
                                      filename='epoch={epoch}-val_loss={validation_step_loss:.5f}')

trainer = pl.Trainer(fast_dev_run=fast_dev_run, logger=logger, max_epochs=1000, accelerator="gpu", devices=[0],
                     callbacks=[early_stopping_callback, checkpoint_callback], deterministic=True,
                     gradient_clip_val=1.0)
trainer.logger._log_graph = True

# %%
trainer.fit(model, datamodule=datamodule)

# %%
with open(checkpoint_path / "best_model.txt", "w") as f:
    f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_model_path}")

trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

# # load the best model and test
# model_path = '/local/home/xintliu/IMUPoser/checkpoints/og_global-07042023-122124/epoch=epoch=198-val_loss=validation_step_loss=0.01176.ckpt'
# model_path = '/local/home/xintliu/IMUPoser/checkpoints/test4_global-07082023-130431/epoch=epoch=198-val_loss=validation_step_loss=0.01222.ckpt'
# model_path = '/local/home/xintliu/IMUPoser/checkpoints/test2_global-07032023-155653/epoch=epoch=217-val_loss=validation_step_loss=0.01120.ckpt'
# bm = model.load_from_checkpoint(model_path)
# bm.eval()
# trainer.test(bm, datamodule)
