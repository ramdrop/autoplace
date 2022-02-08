from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from numpy import linalg as LA
import numpy as np


def plot_pcl(pcls):

    plt.rcParams['figure.figsize'] = (5 * len(pcls), 5)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['savefig.dpi'] = 100

    fig, axes = plt.subplots(1, len(pcls))
    FONTSIZE = 8
    # AXIS_LIMIT = 100
    for i in range(len(pcls)):
        pcl = pcls[i]
        axe = axes[i]
        axe.set_xlabel("x(m)", fontsize=FONTSIZE)
        axe.set_ylabel("y(m)", fontsize=FONTSIZE)
        axe.tick_params(labelsize=FONTSIZE)
        axe.scatter(pcl[:, 0], pcl[:, 1], s=1)
        # axe.set_xlim([-AXIS_LIMIT,AXIS_LIMIT])
        # axe.set_ylim([-AXIS_LIMIT,AXIS_LIMIT])
        axe.set_title('number of points:{}'.format(len(pcl)))
    plt.show()
    plt.savefig('scale_process.jpg')


def normalize_feature(pcl_input, feature_channel=[5, 17], target='-11'):
    # convert to (-1, 1)
    for channel in feature_channel:
        feature = pcl_input[:, channel].reshape(-1, 1)

        # method 1
        scaler = MinMaxScaler()
        scaler.fit(feature)
        feature = scaler.transform(feature)
        if target == '-11':
            feature = 2 * feature - 1
        elif target == '01':
            pass

        # method 2
        # feature = (feature - np.mean(feature)) / np.max(abs(feature - np.mean(feature)))

        pcl_input[:, channel] = feature.flatten()

    return pcl_input


def align(pcl_input, seed, num_points, padding):
    '''
    function:   align point to a fixed number
    input:  (N, D),    N: number of points, D: dimension of a point (each feature has been normalized)
    output: (N, D),    N: number of points, D: dimension of a point
    '''
    SEED = seed
    NUM_POINTS = num_points
    rg = np.random.default_rng(SEED)
    N = pcl_input.shape[0]
    D = pcl_input.shape[1]
    # add random noise (continuous normal distribution (0,1]) to align point number
    if pcl_input.shape[0] < NUM_POINTS:
        if padding == 'drop':
            return None
        elif padding == 'noise':
            if 2 * N < NUM_POINTS:
                return None
            noise = rg.random(size=(NUM_POINTS - N, D), dtype=np.float32)
            noise = 2 * noise - 1   #  scale from (0,1] to (-1,1]
            indices = rg.integers(low=0, high=pcl_input.shape[0], size=1, dtype=np.int32)
            pcl_input = np.insert(pcl_input, indices, noise, axis=0)
        else:
            print('undefined align method')
            exit(0)
    # choose random points
    elif pcl_input.shape[0] > NUM_POINTS:
        indices = rg.choice(pcl_input.shape[0], NUM_POINTS, replace=False)
        pcl_input = pcl_input[indices]

    return pcl_input


def rescale(pcl_input, measure_range):
    '''
    function:   rescale points(x,y,z) to [-1,1]
    input:  (N, D),    N: number of points, D: dimension of a point
    output: (N, D),    N: number of points, D: dimension of a point
    '''
    # remove points outside r=100m
    pcl_dist = LA.norm(pcl_input[:, :3], axis=1)
    far_indices = np.where(pcl_dist > measure_range)
    pcl_near = np.delete(pcl_input, list(far_indices), axis=0)

    # scale points to [-1,1]m
    pcl_near_mean = np.mean(pcl_near[:, :3], axis=0)
    pcl_dist = LA.norm(pcl_near[:, :3] - pcl_near_mean, axis=1)
    # pcl_dist_mean = np.mean(pcl_dist)  # diveded by mean point distance
    pcl_dist_mean = np.max(pcl_dist)               # divided by max point distance
    s = 1.0 / (2 * pcl_dist_mean)
    # s = 1.0/norm_mean
    T = [[s, 0, 0, -s * pcl_near_mean[0]], [0, s, 0, -s * pcl_near_mean[1]], [0, 0, s, -s * pcl_near_mean[2]], [0, 0, 0, 1]]
    pcl_extend = np.hstack((pcl_near[:, :3], np.ones((len(pcl_near), 1))))
    pcl_scaled = np.dot(T, pcl_extend.T).T
    outlierx1 = np.where(pcl_scaled[:, 0] > 1)
    outlierx2 = np.where(pcl_scaled[:, 0] < -1)
    outliery1 = np.where(pcl_scaled[:, 1] > 1)
    outliery2 = np.where(pcl_scaled[:, 1] < -1)
    outlier = set(outlierx1[0])
    outlier.update(set(outlierx2[0]))
    outlier.update(set(outliery1[0]))
    outlier.update(set(outliery2[0]))

    pcl_scaled_feature = np.concatenate((pcl_scaled[:, :3], pcl_near[:, 3:]), axis=1)
    pcl_feature_near = np.delete(pcl_scaled_feature, list(outlier), axis=0)

    return pcl_feature_near
