import torch

from imuposer.config import Config
from imuposer import math
config = Config(experiment='test1')

path_to_save = config.processed_imu_poser_25fps
path_to_save.mkdir(exist_ok=True, parents=True)

# process AMASS first
for fpath in (config.processed_imu_poser / "AMASS").iterdir():
    # resample to 25 fps
    joint = [x for x in torch.load(fpath / "joint.pt")]
    pose = [math.axis_angle_to_rotation_matrix(x.contiguous()).view(-1, 24, 3, 3) for x in torch.load(fpath / "pose.pt")]
    shape = torch.load(fpath / "shape.pt")
    tran = [x for x in torch.load(fpath / "tran.pt")]
    
    # average filter
    vacc = [x for x in torch.load(fpath / "vacc.pt")]
    vrot = [x for x in torch.load(fpath / "vrot.pt")]
    
    # save the data
    fdata = {
        "joint": joint,
        "pose": pose,
        "shape": shape,
        "tran": tran,
        "acc": vacc,
        "ori": vrot
    }
    
    torch.save(fdata, path_to_save / f"{fpath.name}.pt")

# process DIP next
for fpath in (config.processed_imu_poser / "DIP_IMU").iterdir():
    # resample to 25 fps
    joint = [x for x in torch.load(fpath / "joint.pt")]
    pose = [math.axis_angle_to_rotation_matrix(x.contiguous()).view(-1, 24, 3, 3) for x in torch.load(fpath / "pose.pt")]
    shape = torch.load(fpath / "shape.pt")
    tran = [x for x in torch.load(fpath / "tran.pt")]
    
    # average filter
    acc = [x for x in torch.load(fpath / "accs.pt")]
    rot = [x for x in torch.load(fpath / "oris.pt")]
    
    # save the data
    fdata = {
        "joint": joint,
        "pose": pose,
        "shape": shape,
        "tran": tran,
        "acc": acc,
        "ori": rot
    }
    
    torch.save(fdata, path_to_save / f"dip_{fpath.name}.pt")