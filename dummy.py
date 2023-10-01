import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything

from src.imuposer.config import Config, amass_datasets
from src.imuposer import math

config = Config(project_root_dir="../../")
comp_device = torch.device("cuda:" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(comp_device))
print(torch.cuda.current_device())
