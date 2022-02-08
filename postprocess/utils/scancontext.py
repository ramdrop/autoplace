# %%
import os
os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
os.sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import open3d as o3d
import tqdm
import pickle
import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from postprocess.utils.common import load_para, load_split
from postprocess.utils.common import calculate_pr, cal_recall_feat, cal_recall_pred


class ScanContext:
    def __init__(self):
        # static variables
        self.viz = 0
        self.downcell_size = 0.5
        self.sensor_height = 0
        # sector_num = np.array([45, 90, 180, 360, 720])
        # ring_num = np.array([10, 20, 40, 80, 160])
        self.sector_num = 60
        self.ring_num = 20
        self.max_length = 1.414

    def load_velo_scan(self):
        scan = np.fromfile(self.bin_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        ptcloud_xyz = scan[:, :-1]

        return ptcloud_xyz

    def xy2theta(self, x, y):
        if (x >= 0 and y >= 0):
            theta = 180 / np.pi * np.arctan(y / x)
        if (x < 0 and y >= 0):
            theta = 180 - ((180 / np.pi) * np.arctan(y / (-x)))
        if (x < 0 and y < 0):
            theta = 180 + ((180 / np.pi) * np.arctan(y / x))
        if (x >= 0 and y < 0):
            theta = 360 - ((180 / np.pi) * np.arctan((-y) / x))
        # print('x: ', x, 'y: ', y)
        return theta

    def pt2rs(self, point, gap_ring, gap_sector, num_ring, num_sector):
        x = point[0]
        y = point[1]
        # z = point[2]

        if (x == 0.0):
            x = 0.001
        if (y == 0.0):
            y = 0.001

        theta = self.xy2theta(x, y)
        faraway = np.sqrt(x * x + y * y)

        idx_ring = np.divmod(faraway, gap_ring)[0]
        idx_sector = np.divmod(theta, gap_sector)[0]

        if (idx_ring >= num_ring):
            idx_ring = num_ring - 1                                            # python starts with 0 and ends with N-1

        return int(idx_ring), int(idx_sector)

    def ptcloud2sc(self, ptcloud, num_sector, num_ring, max_length):
        num_points = ptcloud.shape[0]

        gap_ring = max_length / num_ring
        gap_sector = 360 / num_sector

        enough_large = 1000
        # divide a circle area into num_ring*num_sector bins,
        # use the max point height to represent each bin
        sc_storage = np.zeros([enough_large, num_ring, num_sector])
        sc_counter = np.zeros([num_ring, num_sector])
        for pt_idx in range(num_points):

            point = ptcloud[pt_idx, :]
            point_height = point[2] + self.sensor_height

            idx_ring, idx_sector = self.pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)

            if sc_counter[idx_ring, idx_sector] >= enough_large:
                continue
            sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
            sc_counter[idx_ring, idx_sector] += 1

        sc = np.amax(sc_storage, axis=0)

        return sc

    def genKey(self, sc):
        return np.count_nonzero(sc, axis=1) / self.sector_num

    def genSCs(self, pcl_file):
        ptcloud_xyz = np.fromfile(pcl_file, dtype=np.float64)
        ptcloud_xyz = ptcloud_xyz.reshape(-1, 4)
        ptcloud_xyz = ptcloud_xyz[:, :3]
        # print("The number of original points: " + str(ptcloud_xyz.shape))

        # downsample pointcloud
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(ptcloud_xyz)
        # downpcd = pcd.voxel_down_sample(voxel_size=self.downcell_size)
        # ptcloud_xyz_downed = np.asarray(downpcd.points)
        # print("The number of downsampled points: " + str(ptcloud_xyz_downed.shape))
        # draw_geometries([downpcd])
        # if (self.viz):
        #     o3d.visualization.draw_geometries([downpcd])

        ptcloud_xyz_downed = ptcloud_xyz
        # for ind, ele in enumerate(ptcloud_xyz_downed[:, 0]):
        #     if np.isnan(ele):
        #         print('x nan: ', ind)
        # for ind, ele in enumerate(ptcloud_xyz_downed[:, 1]):
        #     if np.isnan(ele):
        #         print('y nan: ', ind)
        sc = self.ptcloud2sc(ptcloud_xyz_downed, self.sector_num, self.ring_num, self.max_length)
        return sc
        # scs = []
        # for res in range(len(self.sector_res)):
        #     num_sector = self.sector_res[res]
        #     num_ring = self.ring_res[res]

        #     sc = self.ptcloud2sc(ptcloud_xyz_downed, num_sector, num_ring, self.max_length)
        #     scs.append(sc)
        # return scs

    def plot_multiple_sc(self, fig_idx=1):
        num_res = len(self.sector_res)
        fig, axes = plt.subplots(nrows=num_res)
        axes[0].set_title('Scan Contexts with multiple resolutions', fontsize=14)
        for ax, res in zip(axes, range(num_res)):
            ax.imshow(self.SCs[res])
        plt.show()

    def distance_sc(self, sc1, sc2):
        num_sectors = sc1.shape[1]
        # repeate to move 1 columns
        sim_for_each_cols = np.zeros(num_sectors)

        for i in range(num_sectors):
            # Shift
            one_step = 1                                                           # const
            sc1 = np.roll(sc1, one_step, axis=1)                                   #  columne shift

            # compare
            sum_of_cos_sim = 0
            num_col_engaged = 0

            for j in range(num_sectors):
                col_j_1 = sc1[:, j]
                col_j_2 = sc2[:, j]

                if (~np.any(col_j_1) or ~np.any(col_j_2)):
                    continue

                # calc sim
                cos_similarity = np.dot(col_j_1, col_j_2) / (np.linalg.norm(col_j_1) * np.linalg.norm(col_j_2))
                sum_of_cos_sim = sum_of_cos_sim + cos_similarity

                num_col_engaged = num_col_engaged + 1

            # devided by num_col_engaged: So, even if there are many columns that are excluded from the calculation, we
            # can get high scores if other columns are well fit.
            sim_for_each_cols[i] = sum_of_cos_sim / num_col_engaged

        sim = max(sim_for_each_cols)

        dist = 1 - sim

        return dist


# %% Sanity Check
# PCL_DIR = '/LOCAL/ramdrop/dataset/nuscenes_radar/imgs_b/pcl'
# pcl_list = os.listdir(PCL_DIR)
# pcl_list.sort(key=lambda x: int(x[:5]))
# sc_manager = ScanContext()
# for index, pcl_file in enumerate(tqdm.tqdm(pcl_list, ncols=40)):
#     sc = sc_manager.genSCs(os.path.join(PCL_DIR, pcl_file))
#     dist = sc_manager.distance_sc(sc, sc)
#     print('sc:', sc.shape)
#     print('dist:', dist)
# pass
