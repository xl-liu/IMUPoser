r"""
    Evaluate the pose estimation.
    from TransPose
"""

import torch
import pytorch_lightning as pl
import tqdm
import os
from imuposer.math.angular import *
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.config import Config
import numpy as np

config = Config()
# where the sensors are attached
tracking_sensors = [18, 19, 4, 5, 15, 0]
sip_eval_sensors = [1, 2, 16, 17]

# the remaining "sensors" are evaluation sensors
all_sensors = [*range(24)]
remaining_eval_sensors = [s for s in all_sensors if s not in tracking_sensors and s not in sip_eval_sensors]

def normalize_and_concat(glb_acc, glb_ori):
    glb_acc = glb_acc.view(-1, 6, 3)
    glb_ori = glb_ori.view(-1, 6, 3, 3)
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1]) 
    ori = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
    return data

class BasePoseEvaluator:
    r"""
    Base class for evaluators that evaluate motions.
    """
    def __init__(self, official_model_file: str, rep=RotationRepresentation.ROTATION_MATRIX, use_pose_blendshape=False,
                 device=torch.device('cpu')):
        self.model = ParametricModel(official_model_file, use_pose_blendshape=use_pose_blendshape, device=device)
        self.rep = rep
        self.device = device

    def _preprocess(self, pose, shape=None, tran=None):
        pose = to_rotation_matrix(pose.to(self.device), self.rep).view(pose.shape[0], -1)
        shape = shape.to(self.device) if shape is not None else shape
        tran = tran.to(self.device) if tran is not None else tran
        return pose, shape, tran
    
class FullMotionEvaluator(BasePoseEvaluator):
    r"""
    Evaluator for full motions (pose sequences with global translations). Plenty of metrics.
    """
    def __init__(self, official_model_file: str, align_joint=None, rep=RotationRepresentation.ROTATION_MATRIX,
                 use_pose_blendshape=False, fps=60, joint_mask=None, device=torch.device('cpu')):
        r"""
        Init a full motion evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param use_pose_blendshape: Whether to use pose blendshape or not.
        :param joint_mask: If not None, local angle error, global angle error, and joint position error
                           for these joints will be calculated additionally.
        :param fps: Motion fps, by default 60.
        :param device: torch.device, cpu or cuda.
        """
        super(FullMotionEvaluator, self).__init__(official_model_file, rep, use_pose_blendshape, device=device)
        self.align_joint = 0 if align_joint is None else align_joint.value
        self.fps = fps
        self.joint_mask = joint_mask

    def __call__(self, pose_p, pose_t, shape_p=None, shape_t=None, tran_p=None, tran_t=None):
        r"""
        Get the measured errors. The returned tensor in shape [10, 2] contains mean and std of:
          0.  Joint position error (align_joint position aligned).
          1.  Vertex position error (align_joint position aligned).
          2.  Joint local angle error (in degrees).
          3.  Joint global angle error (in degrees).
          4.  Predicted motion jerk (with global translation).
          5.  True motion jerk (with global translation).
          6.  Translation error (mean root translation error per second, using a time window size of 1s).
          7.  Masked joint position error (align_joint position aligned, zero if mask is None).
          8.  Masked joint local angle error. (in degrees, zero if mask is None).
          9.  Masked joint global angle error. (in degrees, zero if mask is None).

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param shape_p: Predicted shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param shape_t: True shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran_p: Predicted translations in shape [batch_size, 3]. Use None for zeros.
        :param tran_t: True translations in shape [batch_size, 3]. Use None for zeros.
        :return: Tensor in shape [10, 2] for the mean and std of all errors.
        """
        f = self.fps
        pose_local_p, shape_p, tran_p = self._preprocess(pose_p, shape_p, tran_p)
        pose_local_t, shape_t, tran_t = self._preprocess(pose_t, shape_t, tran_t)
        pose_global_p, joint_p, vertex_p = self.model.forward_kinematics(pose_local_p, shape_p, tran_p, calc_mesh=True)
        pose_global_t, joint_t, vertex_t = self.model.forward_kinematics(pose_local_t, shape_t, tran_t, calc_mesh=True)

        offset_from_p_to_t = (joint_t[:, self.align_joint] - joint_p[:, self.align_joint]).unsqueeze(1)
        ve = (vertex_p + offset_from_p_to_t - vertex_t).norm(dim=2)   # N, J
        je = (joint_p + offset_from_p_to_t - joint_t).norm(dim=2)     # N, J
        lae = radian_to_degree(angle_between(pose_local_p, pose_local_t).view(pose_p.shape[0], -1))           # N, J
        gae = radian_to_degree(angle_between(pose_global_p, pose_global_t).view(pose_p.shape[0], -1))         # N, J
        jkp = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2)   # N, J
        jkt = ((joint_t[3:] - 3 * joint_t[2:-1] + 3 * joint_t[1:-2] - joint_t[:-3]) * (f ** 3)).norm(dim=2)   # N, J
        te = ((joint_p[f:, :1] - joint_p[:-f, :1]) - (joint_t[f:, :1] - joint_t[:-f, :1])).norm(dim=2)        # N, 1
        mje = je[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)     # N, mJ
        mlae = lae[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)   # N, mJ
        mgae = gae[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)   # N, mJ

        return torch.tensor([[je.mean(),   je.std(dim=0).mean()],
                             [ve.mean(),   ve.std(dim=0).mean()],
                             [lae.mean(),  lae.std(dim=0).mean()],
                             [gae.mean(),  gae.std(dim=0).mean()],
                             [jkp.mean(),  jkp.std(dim=0).mean()],
                             [jkt.mean(),  jkt.std(dim=0).mean()],
                             [te.mean(),   te.std(dim=0).mean()],
                             [mje.mean(),  mje.std(dim=0).mean()],
                             [mlae.mean(), mlae.std(dim=0).mean()],
                             [mgae.mean(), mgae.std(dim=0).mean()]])


class PoseEvaluator:
    def __init__(self):
        self._eval_fn = FullMotionEvaluator(config.og_smpl_model_path, joint_mask=torch.tensor([1, 2, 16, 17]), fps=config.target_fps)
        # sip_eval_sensors = [1, 2, 16, 17]
    def eval(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        # print("----------------------pose shape", pose_p.shape)
        # print("----------------------target shape", pose_t.shape)
        # ignored_joints = [0, 7, 8, 10, 11, 20, 21, 22, 23]      # transpose
        # pa = pose_p[:, ignored_joints]
        # print("-----------------------ignored", pa.shape)
        # pb = torch.eye(3, device=pose_p.device)
        # print("-----------------------identity", pb.shape)
        # pose_p[:, ignored_joints] = torch.eye(3, device=pose_p.device)
        # pose_t[:, ignored_joints] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))

def evaluate_pose(dataset, num_past_frame=20, num_future_frame=5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    evaluator = PoseEvaluator()
    

    net = TransPoseNet(num_past_frame, num_future_frame).to(device)
    data = torch.load(os.path.join(dataset, 'test.pt'))
    xs = [normalize_and_concat(a, r).to(device) for a, r in zip(data['acc'], data['ori'])]
    ys = [(axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3), t) for p, t in zip(data['pose'], data['tran'])]
    offline_errs, online_errs = [], []
    for x, y in tqdm.tqdm(list(zip(xs, ys))):
        net.reset()
        online_results = [net.forward_online(f) for f in torch.cat((x, x[-1].repeat(num_future_frame, 1)))]
        pose_p_online, tran_p_online = [torch.stack(_)[num_future_frame:] for _ in zip(*online_results)]
        pose_p_offline, tran_p_offline = net.forward_offline(x)
        pose_t, tran_t = y
        offline_errs.append(evaluator.eval(pose_p_offline, pose_t))
        online_errs.append(evaluator.eval(pose_p_online, pose_t))
    print('============== offline ================')
    evaluator.print(torch.stack(offline_errs).mean(dim=0))
    print('============== online ================')
    evaluator.print(torch.stack(online_errs).mean(dim=0))


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False   # if cudnn error, uncomment this line
    evaluate_pose(config.processed_imu_poser_new / 'dip_test.py')
    # evaluate_pose(paths.totalcapture_dir)
