r"""
    Preprocess DIP-IMU and TotalCapture test dataset.
    Synthesize AMASS dataset.

"""

import torch
from imuposer.config import Config, amass_datasets
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math
import os
import pickle
import numpy as np
from tqdm import tqdm
import glob
from scipy.interpolate import interp1d
from scipy.signal import firwin
from scipy import signal
# import torchaudio.transforms as T
import torch.nn.functional as F
from noise import NoiseGenerator
# TODO: fix torchaudio

# config = Config(project_root_dir="../../")
# config = Config(experiment='test1')
config = Config()
# path_to_save = config.processed_imu_poser_new
path_to_save = config.processed_imu_poser_noisy
path_to_save.mkdir(exist_ok=True, parents=True)

TARGET_FPS = config.target_fps
# DIP masks
# VI_MASK = [1962, 5431, 1096, 4583, 412, 3021]
VI_MASK = [1961, 5424, 1176, 4662, 411, 3021] # transpose
JI_MASK = [18, 19, 4, 5, 15, 0]

# IMUPoser masks
# left wrist, right wrist, left thigh, right thigh, head, pelvis
# vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
# ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

def get_acc_fd(v, noise=False):
    r"""
    fd accelerations from vertex positions.
    """
    n = len(v)
    h = 1/TARGET_FPS
    # reshape the input 
    v = v.reshape(-1, 18)   # 6x3
    # v = v.detach().numpy()

    # filter the trajectory
    b = firwin(5, 0.2)
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

    if noise:
        acc = add_noise(acc)

    return acc

def add_noise(acc, ori=None):
    """
    add noise to the synthetic imu data based on total capture imu's allan variance 
    """
    ng = NoiseGenerator()
    dt = 1.0/TARGET_FPS
    l = acc.shape[0]
    for i in range(3):
        n = config.acc_random_walk[i]
        b = config.acc_bias[i]
        for j in range(6):
            noise = ng.generate(dt, l, colour=ng.white(n)) + ng.generate(dt, l, colour=ng.pink(b))
            # noise is always of even length
            if len(noise) != l:
                noise = np.append(noise, 0)
            acc[:,j,i] += noise

    if ori is not None:
        n = config.ori_random_walk[i]
        b = config.ori_bias[i]
        ori_out = ori.clone()
        return acc, ori_out
    else:
        return acc

def process_accel(vertices, ori):
    # get global acceleration from the vertex positions
    acc = get_acc_fd(vertices)

    # add gravity
    acc += np.array([0, 9.8707, 0])

    # convert to local frame
    acc = torch.matmul(ori, acc)

    # add noise
    acc, ori_out = add_noise(acc, ori)

    # convert back to global frame 

    # remove gravity

    return acc_out, ori_out

def resample(data_in, fps_in=60, fps_out=TARGET_FPS):
    """
    interpolate the orignal poses to the target datarate 
    poses: nx156
    """
    # TODO: use better spline methods for rotation (quaternion k spline?)
    poses_out = []
    n_frames = np.shape(data_in)[0]
    t_end = n_frames/fps_in
    t_in = np.linspace(0, t_end, n_frames)
    t_out = np.linspace(0, t_end, int(t_end*fps_out))
    
    interp = interp1d(t_in, data_in, axis=0)
    poses_out = interp(t_out)
    if type(data_in) == torch.Tensor:
        poses_out = torch.tensor(poses_out, dtype=torch.float)

    return poses_out

def smooth_avg(acc=None, s=3):
    nan_tensor = (torch.zeros((s // 2, acc.shape[1], acc.shape[2])) * torch.nan)
    acc = torch.cat((nan_tensor, acc, nan_tensor))
    tensors = []
    for i in range(s):
        L = acc.shape[0]
        tensors.append(acc[i:L-(s-i-1)])

    smoothed = torch.stack(tensors).nanmean(dim=0)
    return smoothed

def process_amass(body_model=ParametricModel(config.og_smpl_model_path)):
    # TODO: add and subtract gravity
    # TODO: use different models for different genders

    try:
        processed = [fpath.name for fpath in config.processed_imu_poser_25fps.iterdir()]
    except:
        processed = []    

    for ds_name in amass_datasets:
        if ds_name in processed:
            continue
        data_pose, data_trans, data_beta, length = [], [], [], []
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(config.raw_amass_path, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if cdata['poses'].shape[0] <= 20:
                print('\tdiscard one sequence with length', cdata['poses'].shape[0])
                continue
            if framerate != TARGET_FPS:
                poses = resample(cdata['poses'], framerate, TARGET_FPS)
                trans = resample(cdata['trans'], framerate, TARGET_FPS)
                # resampler = T.Resample(framerate, TARGET_FPS, dtype=np.float32)
                # poses = resampler(poses)
                # trans = resampler(trans)
            data_pose.extend(poses)
            data_trans.extend(trans)
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
            if l <= 20: b += l; print('\tdiscard one sequence with length', l); continue
            p = math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
            out_pose.append(p.clone())  # N, 24, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            # out_vacc.append(_syn_acc(vert[:, VI_MASK]))  # N, 6, 3
            vacc = get_acc_fd(vert[:, VI_MASK], noise=False) # add noise
            out_vacc.append(smooth_avg(vacc,5))   # smooth
            out_vrot.append(grot[:, JI_MASK])  # N, 6, 3, 3
            b += l

        fdata = {
            "joint": out_joint,
            "pose": out_pose,
            "shape": out_shape,
            "tran": out_tran,
            "acc": out_vacc,
            "ori": out_vrot
        }
        torch.save(fdata, path_to_save / f"{ds_name}.pt")
        print('Synthetic AMASS dataset is saved at', path_to_save)
    
def process_dipimu(split="test", body_model=ParametricModel(config.og_smpl_model_path)):
    # resampler = T.Resample(60, TARGET_FPS, dtype=np.float32)
    out_joint, out_pose, out_shape, out_tran, out_acc, out_rot = [], [], [], [], [], []
    imu_mask = [7, 8, 9, 10, 0, 2]
    if split == "test":
        test_split = ['s_09', 's_10']
    else:
        test_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    accs, oris, poses, trans, shapes, joints, vrots, vaccs = [], [], [], [], [], [], [], []
    
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
                accs.append(resample(smooth_avg(acc.clone()), 5))
                oris.append(resample(ori.clone()))
                trans.append(resample(tran.clone()))
                shapes.append(resample(shape.clone())) # default shape
                
                # forward kinematics to get the joint position
                p = math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                poses.append(resample(p.clone()))
                grot, joint, vert = body_model.forward_kinematics(p, shape, tran, calc_mesh=True)
                # vacc = get_acc_fd(vert[:, VI_MASK])
                # vrot = grot[:, JI_MASK]
                joints.append(resample(joint.clone()))
                # vaccs.append(vacc)
                # vrots.append(vrot)
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))
        
        out_joint.extend(joints)
        out_pose.extend(poses)
        out_shape.extend(shapes)
        out_tran.extend(trans)
        out_acc.extend(accs)
        out_rot.extend(oris)

    # save the data
    fdata = {
        "joint": out_joint,
        "pose": out_pose,
        "shape": out_shape,
        "tran": out_tran,
        "acc": out_acc,
        "ori": out_rot
    }
    torch.save(fdata, path_to_save / f"dip_{split}.pt")
    print('Preprocessed DIP-IMU dataset is saved at', path_to_save)

def process_totalcapture(split="test", body_model=ParametricModel(config.og_smpl_model_path)):
    inches_to_meters = 0.0254
    imu_mask = [5, 6, 7, 8, 0, 2]
    raw_tc_path = config.raw_totalcapture_path

    if split == "test":
        test_split = [x.name for x in raw_tc_path.iterdir() if "s1" in x.name or "s2" in x.name]
    else:
        test_split = [x.name for x in raw_tc_path.iterdir() if "s1" not in x.name]
    accs, oris, poses, trans, shapes, joints, vrots, vaccs = [], [], [], [], [], [], [], []
    
    for action_sequence in test_split:
        path = os.path.join(raw_tc_path, action_sequence)
        data = pickle.load(open(path, 'rb'), encoding='latin1')

        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        p = math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        poses.append(p)  # N, 24, 3

        shapes.append(torch.ones((10)))
        trans.append(torch.zeros(pose.shape[0], 3)) # tc does not contain translations

        # forward kinematics to get the joint position
        # grot, joint, vert = body_model.forward_kinematcs(p, torch.ones((10)), torch.zeros(pose.shape[0], 3), calc_mesh=True)
        # joints.append((joint.clone()))

        # for iacc, pose in zip(accs, poses):
        #     pose = math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        #     _, _, vert = body_model.forward_kinematics(pose, calc_mesh=True)
        #     vacc = get_acc_fd(vert[:, VI_MASK])
        #     for imu_id in range(6):
        #         for i in range(3):
        #             d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
        #             iacc[:, imu_id, i] += d

        out_acc = [resample(smooth_avg(acc, 5)) for acc in accs] 
        out_ori = [resample(ori) for ori in oris]   
        out_pose = [resample(pose) for pose in poses]  
        
    # save the data
    fdata = {
        # "joint": joints,
        "pose": out_pose,
        "shape": shapes,
        # "tran": trans,
        "acc": out_acc,
        "ori": out_ori
    }
    torch.save(fdata, path_to_save / f"total_capture_{split}.pt")
    print('Preprocessed TotalCapture dataset is saved at', path_to_save)


if __name__ == '__main__':
    # process_dipimu(split="test")
    # process_dipimu(split="train")
    # process_amass()
    process_totalcapture("test")
