r"""
    Preprocess DIP-IMU and TotalCapture test dataset.
    Synthesize AMASS dataset.

"""

# %load_ext autoreload
# %autoreload 2

import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
import glob
from scipy.interpolate import splrep, splev

from imuposer.config import Config, amass_datasets
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math

from scipy.signal import firwin
from scipy import signal

# config = Config(project_root_dir="../../")
config = Config(experiment='test1')

TARGET_FPS = 25
# DIP masks
# VI_MASK = [1962, 5431, 1096, 4583, 412, 3021]
VI_MASK = [1961, 5424, 1176, 4662, 411, 3021] # transpose
JI_MASK = [18, 19, 4, 5, 15, 0]

# IMUPoser masks
# left wrist, right wrist, left thigh, right thigh, head, pelvis
# vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
# ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

def get_acc_fd(v):
    r"""
    fd accelerations from vertex positions.
    """
    n = len(v)
    h = 1/TARGET_FPS
    # reshape the input 
    v = v.reshape(-1, 18)   # 6x3
    # v = v.detach().numpy()

    # filter the trajectory
    b = firwin(10, 0.2)
    a = 1
    f = signal.filtfilt(b, a, v, axis=0)

    acc = np.zeros(np.shape(f))
    for i in range(4, n-4):
        acc[i,:] = (1/(h**2)) * (-1/560*f[i+4,:] + 8/315*f[i+3,:] - 1/5*f[i+2,:] + 8/5*f[i+1,:] - 205/72*f[i,:] 
                            + 8/5*f[i-1,:] - 1/5*f[i-2,:] + 8/315*f[i-3,:] - 1/560*f[i-4,:])
    acc[:4,:] = acc[4,:]
    acc[-4:,:] = acc[-5,:]
    # reshape the output 
    acc = torch.reshape(torch.tensor(acc, dtype=torch.float), (-1, 6, 3))

    return acc

def interpolate_poses(poses, fps_in, fps_out):
    """
    interpolate the orignal poses to the target datarate 
    poses: nx156
    """

    # TODO: use quaternion spline 

    poses_out = []
    n_frames = np.shape(poses)[0]
    t_in = np.linspace(0, n_frames/fps_in, n_frames)
    t_out = np.linspace(0, n_frames/fps_out, n_frames)
    n = np.shape(poses)[1]
    
    for i in range(n):
        spl = splrep(t_in, poses[:, i])
        poses_out.append(splev(t_out, spl))

    poses_out = np.array(poses_out).T

    return poses_out

def process_amass():
    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        return acc

    body_model = ParametricModel(config.og_smpl_model_path)

    try:
        processed = [fpath.name for fpath in (config.processed_imu_poser / "AMASS").iterdir()]
    except:
        processed = []
    # processed = []

    for ds_name in amass_datasets:
        if ds_name in processed:
            continue
        data_pose, data_trans, data_beta, length = [], [], [], []
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(config.raw_amass_path, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if cdata['poses'].shape[0] <=10:
                continue
            if framerate != TARGET_FPS:
                poses = interpolate_poses(cdata['poses'], framerate, TARGET_FPS)
                trans = interpolate_poses(cdata['trans'], framerate, TARGET_FPS)
            
            data_pose.extend(poses.astype(np.float32))
            data_trans.extend(trans.astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(poses.shape[0])

        if len(data_pose) == 0:
            print(f"AMASS dataset, {ds_name} not supported")
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)

        # include the left and right index fingers in the pose
        pose[:, 23] = pose[:, 37]     # right hand 
        pose = pose[:, :24].clone()   # only use body + right and left fingers

        # align AMASS global frame with DIP
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0])))

        print('Synthesizing IMU accelerations and orientations')
        b = 0
        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
        for i, l in tqdm(list(enumerate(length))):
            if l <= 30: b += l; print('\tdiscard one sequence with length', l); continue
            p = math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
            out_pose.append(pose[b:b + l].clone())  # N, 24, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            # out_vacc.append(_syn_acc(vert[:, VI_MASK]))  # N, 6, 3
            # TODO:
            tmp_acc = get_acc_fd(vert[:, VI_MASK])
            out_vacc.append(tmp_acc)
            out_vrot.append(grot[:, JI_MASK])  # N, 6, 3, 3
            b += l

        print('Saving')
        amass_dir = config.processed_imu_poser / "AMASS"
        amass_dir.mkdir(exist_ok=True, parents=True)
        ds_dir = amass_dir / ds_name
        ds_dir.mkdir(exist_ok=True)

        torch.save(out_pose, ds_dir / 'pose.pt')
        torch.save(out_shape, ds_dir / 'shape.pt')
        torch.save(out_tran, ds_dir / 'tran.pt')
        torch.save(out_joint, ds_dir / 'joint.pt')
        torch.save(out_vrot, ds_dir / 'vrot.pt')
        torch.save(out_vacc, ds_dir / 'vacc.pt')
        print('Synthetic AMASS dataset is saved at', str(ds_dir))

def process_dipimu(split="test"):
    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        return acc
    
    imu_mask = [7, 8, 9, 10, 0, 2]
    if split == "test":
        test_split = ['s_09', 's_10']
    else:
        test_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    accs, oris, poses, trans, shapes, joints, vrots, vaccs = [], [], [], [], [], [], [], []
    
    body_model = ParametricModel(config.og_smpl_model_path)

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(config.raw_dip_path, subject_name)):
            path = os.path.join(config.raw_dip_path, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            shape = torch.ones((10))
            tran = torch.zeros(pose.shape[0], 3) # dip-imu does not contain translations
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(tran.clone())  
                
                shapes.append(shape.clone()) # default shape
                
                # forward kinematics to get the joint position
                p = math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                grot, joint, vert = body_model.forward_kinematics(p, shape, tran, calc_mesh=True)
                vacc = get_acc_fd(vert[:, VI_MASK])
                vrot = grot[:, JI_MASK]
                
                joints.append(joint)
                vaccs.append(vacc)
                vrots.append(vrot)
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))
                
    path_to_save = config.processed_imu_poser / f"DIP_IMU/{split}"
    path_to_save.mkdir(exist_ok=True, parents=True)
    
    torch.save(poses, path_to_save / 'pose.pt')
    torch.save(shapes, path_to_save / 'shape.pt')
    torch.save(trans, path_to_save / 'tran.pt')
    torch.save(joints, path_to_save / 'joint.pt')
    torch.save(vrots, path_to_save / 'vrot.pt')
    torch.save(vaccs, path_to_save / 'vacc.pt')
    torch.save(oris, path_to_save / 'oris.pt')
    torch.save(accs, path_to_save / 'accs.pt')
    
    print('Preprocessed DIP-IMU dataset is saved at', path_to_save)

if __name__ == '__main__':
    # process_dipimu(split="test")
    # process_dipimu(split="train")
    process_amass()
