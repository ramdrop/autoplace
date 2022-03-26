# -*- coding: utf-8 -*-
"""
general operation

"""

from matplotlib.pyplot import axis
from nuscenes import NuScenes
import os.path as osp
from nuscenes.utils.data_classes import RadarPointCloud
import numpy as np
from scipy.optimize import least_squares
import os
os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.ransac_range_v import RANSAC_Solver
from ipdb import set_trace
import socket


# Name of the location:
# "singapore-onenorth"
# "singapore-hollandvillage"
# "singapore-queenstown"
# "boston-seaport"


class Generic(object):
    def __init__(self, split):
        self.dataset = ''   # remember to define your official nuscenes dataset directory here
        assert self.dataset != '', 'the nuscenes dataset directory definition is missing.'
        print(split, 'loading...')
        self.nusc = NuScenes(version=split, dataroot=self.dataset, verbose=False)
        print('done!')
        self.scene = self.nusc.scene[0]
        self.sample_token = self.scene['first_sample_token']
        self.sample = self.nusc.get('sample', self.sample_token)
        self.channel = 'RADAR_FRONT'
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][self.channel])
        self.cs = self.nusc.get('calibrated_sensor', self.sample_data['calibrated_sensor_token'])
        ego_pose_record = self.nusc.get('ego_pose', self.sample_data['ego_pose_token'])
        self.sample_abs_ego_pose = ego_pose_record['translation']
        self.ransac_solver = RANSAC_Solver(0.15, 10)

    def to_next_sample(self):
        self.sample_token = self.sample['next']
        if self.sample_token != '':
            self.sample = self.nusc.get('sample', self.sample_token)

    def to_scene(self, i):
        self.scene = self.nusc.scene[i]
        self.sample_token = self.scene['first_sample_token']
        self.sample = self.nusc.get('sample', self.sample_token)

    def to_sample(self, i):
        if (i > self.scene['nbr_samples']):
            print('index over scene length')
            return
        for _ in range(i):
            self.to_next_sample()

    def get_timestamp(self):
        return self.sample['timestamp']

    def get_sample_data(self, channel='RADAR_FRONT'):
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][channel])
        return self.sample_data

    def get_sample_abs_ego_pose(self, channel='RADAR_FRONT'):
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][self.channel])
        ego_pose_record = self.nusc.get('ego_pose', self.sample_data['ego_pose_token'])
        self.sample_abs_ego_pose = ego_pose_record['translation']
        return self.sample_abs_ego_pose

    def get_pt(self, channel):
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][channel])
        pcl_file = osp.join(self.dataset, self.sample_data['filename'])
        pcl = RadarPointCloud.from_file(pcl_file)
        pcl = pcl.points.transpose()
        return pcl

    def get_pcl(self, channel):
        self.sample_data = self.nusc.get('sample_data', self.sample['data'][channel])
        pcl_file = osp.join(self.dataset, self.sample_data['filename'])
        pcl = RadarPointCloud.from_file(pcl_file)
        pcl = pcl.points[:3, :].transpose()
        return pcl

    def get_pcl_pano(self, chan='RADAR_FRONT', ref_chan='RADAR_FRONT', nsweeps=5):
        ref_chan = 'RADAR_FRONT'
        chan = 'RADAR_FRONT'
        chans = ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']
        pcl_all_ = np.zeros((0, 18))
        for chan in chans:
            pc, times, _ = RadarPointCloud.from_file_multisweep(nusc=self.nusc, sample_rec=self.sample, nsweeps=nsweeps, chan=chan, ref_chan=ref_chan) # (1, 71)
            pt = pc.points[:17, :].transpose()                                                     # (71, 17)
            pt = np.hstack((pt, times.transpose()))                                                # (71, 18)
            pcl_all_ = np.concatenate((pcl_all_, pt), axis=0)

        return pcl_all_

    def get_pcl_pano_filtered(self, info, chan='RADAR_FRONT', ref_chan='RADAR_FRONT', nsweeps=5):
        ref_chan = 'RADAR_FRONT'
        chan = 'RADAR_FRONT'
        chans = ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']
        pcl_all_ = np.zeros((0, 18))
        for chan in chans:
            pcls, times, nbr_points = RadarPointCloud.from_file_multisweep(nusc=self.nusc, sample_rec=self.sample, nsweeps=nsweeps, chan=chan, ref_chan=ref_chan)
            pcls = pcls.points.transpose()  # (53, 18)

            # -------- generate stationary mask for a pointcloud based on doppler velocity ------- #
            if chan not in ['RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']:
                nbr_flag = np.cumsum(nbr_points)
                pcl_list = np.split(pcls, nbr_flag, axis=0)
                pcls_new = np.zeros((0, 18))
                for index, pcl in enumerate(pcl_list[:-1]):
                    best_mask, _, _ = self.ransac_solver.ransac_nusc([info, chan, index], pcl, vis=False)
                    if best_mask is not None:
                        pcl = pcl[best_mask]
                    pcls_new = np.vstack((pcls_new, pcl))
                pcls = pcls_new

            pcl_all_ = np.concatenate((pcl_all_, pcls), axis=0)

        return pcl_all_

    def get_pcl_pano_filtered_mask(self, info, chan='RADAR_FRONT', ref_chan='RADAR_FRONT', nsweeps=5):
        ref_chan = 'RADAR_FRONT'
        chan = 'RADAR_FRONT'
        # chans = ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']
        chans = ['RADAR_FRONT']
        pcl_all_ = np.zeros((0, 19))
        for chan in chans:
            pcls, times, nbr_points = RadarPointCloud.from_file_multisweep(nusc=self.nusc, sample_rec=self.sample, nsweeps=nsweeps, chan=chan, ref_chan=ref_chan)
            pcls = pcls.points.transpose()  # (53, 18)

            # -------- generate stationary mask for a pointcloud based on doppler velocity ------- #
            if chan in ['RADAR_FRONT']:
                nbr_flag = np.cumsum(nbr_points)
                pcl_list = np.split(pcls, nbr_flag, axis=0)
                pcls_new = np.zeros((0, 18 + 1))
                for index, pcl in enumerate(pcl_list[:-1]):
                    best_mask, _, _ = self.ransac_solver.ransac_nusc([info, chan, index], pcl, vis=True)
                    if best_mask is not None:
                        # pcl = pcl[best_mask]
                        pcl = np.hstack((pcl, best_mask.reshape(-1, 1)))
                    pcls_new = np.vstack((pcls_new, pcl))
                pcls = pcls_new
            else:
                pcls = np.hstack((pcls, np.ones((pcls.shape[0], 1))))

            pcl_all_ = np.concatenate((pcl_all_, pcls), axis=0)

        return pcl_all_

    def get_location_indices(self, location):
        boston_indices = []
        for scene_index in range(len(self.nusc.scene)):
            self.to_scene(scene_index)
            if self.nusc.get('log', self.scene['log_token'])['location'] != location:
                continue
            boston_indices.append(scene_index)

        return np.array(boston_indices)
