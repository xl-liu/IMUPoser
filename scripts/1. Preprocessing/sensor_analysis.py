import numpy as np
import allantools
import matplotlib.pyplot as plt
import os
import pickle
from scipy.optimize import nnls

# resampler = T.Resample(60, TARGET_FPS, dtype=np.float32)
# imu_mask = [7, 8, 9, 10, 0, 2]
imu_mask = 10

subjects = ['s_01']
motions = ['01']
accs, oris, poses, trans, shapes, joints, vrots, vaccs = [], [], [], [], [], [], [], []
raw_dip_path = '../data/DIP_IMU_and_Others/DIP_IMU/'    # called from the imuposer directory
labels = ['x', 'y', 'z']

def params_from_avar(tau, avar, effects=None):
    r"""Estimate noise parameters from Allan variance.
    from https://github.com/nmayorov/allan-variance

    The parameters being estimated are typical for inertial sensors:
    quantization noise, additive white noise, flicker noise (long term bias
    instability), random walk and linear ramp (this is a deterministic effect).

    The parameters are estimated using linear least squares with weights
    inversely proportional to the values of Allan variance. That is the sum of
    relative error is minimized. This approach is approximately equivalent of
    doing estimation in the log-log scale.

    Parameters
    ----------
    tau : ndarray, shape (n,)
        Values of averaging time.
    avar : ndarray, shape (n,) or (n, m)
        Values of Allan variance corresponding to `tau`.
    effects : list or None, optional
        Which effects to estimate. Allowed effects are 'quantization', 'white',
        'flicker', 'walk', 'ramp'. If None (default), estimate all of the
        mentioned above effects.

    Returns
    -------
    params : pandas DataFrame or Series
        Estimated parameters.
    """
    ALLOWED_EFFECTS = ['quantization', 'random_walk', 'bias_instability', 'rate_random_walk', 'rate_ramp']

    avar = np.asarray(avar)

    if effects is None:
        effects = ALLOWED_EFFECTS
    elif not set(effects) <= set(ALLOWED_EFFECTS):
        raise ValueError("Unknown effects are passed.")
    n = len(tau)

    A = np.empty((n, 5))
    A[:, 0] = 3 / tau**2
    A[:, 1] = 1 / tau
    A[:, 2] = 2 * np.log(2) / np.pi
    A[:, 3] = tau / 3
    A[:, 4] = tau**2 / 2
    mask = ['quantization' in effects,
            'random_walk' in effects,
            'bias_instability' in effects,
            'rate_random_walk' in effects,
            'rate_ramp' in effects]

    A = A[:, mask]
    effects = np.asarray(ALLOWED_EFFECTS)[mask]

    params = []

    A_scaled = A / avar[:, None]
    x = nnls(A_scaled, np.ones(n))[0]
    params = np.sqrt(x)

    params_dict = {effects[i]:params[i] for i in range(5)}

    return params_dict

for subject_name in subjects:
    # for motion_name in os.listdir(os.path.join(raw_dip_path, subject_name)):
    for motion_name in motions:
        path = os.path.join(raw_dip_path, subject_name, motion_name) + '.pkl'
        data = pickle.load(open(path, 'rb'), encoding='latin1')
        acc = data['imu_acc'][:, imu_mask].reshape((-1, 3))[:150,:]
        if np.isnan(acc).sum() == 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[9, 20])
            for i in range(3):
                a = acc[:,i]
                (taus, adevs, errors, ns) = allantools.oadev(a, rate=60)
                par = params_from_avar(taus, adevs, effects=None)
                print(par)
                ax1.plot(a, label=labels[i])
                ax2.loglog(taus, adevs, label=labels[i])
            plt.legend()
            fig.savefig(f"acc_{subject_name}_{motion_name}_imu_{imu_mask}.png")
        else:
            print('nans')
