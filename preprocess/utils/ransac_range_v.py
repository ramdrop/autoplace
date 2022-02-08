from scipy.optimize import least_squares
import numpy as np
import os
from matplotlib import pyplot as plt
from ipdb import set_trace

np.random.seed(12345)

class RANSAC_Solver:
    def __init__(self, threshold=0.05, max_iter=20) -> None:
        self.threshold = threshold
        self.max_iter = max_iter

    def least_square_solver(self, theta, vr):
        '''
        input:  theta - target azimuth
                vr - target range velocity
        output: stationary mask
        '''
        def err(p, theta, vr):
            alpha = p[0]
            vs = p[1]
            error = vs * np.cos(alpha - theta) - vr
            return error

        p0 = [0, 3]                                                            # alpha, vs
        ret = least_squares(err, p0, args=(theta, vr), verbose=0)
        alpha_ = ret['x'][0]
        vs_ = ret['x'][1]
        return alpha_, vs_

    def ransac(self, theta, vr):
        max_nbr_inliers = 0
        best_alpha_pre = -1
        best_vs_pre = -1
        best_mask = []
        for i in range(self.max_iter):
            inds = np.random.choice(len(vr) - 1, 5)
            # set_trace()
            alpha_pre, vs_pre = self.least_square_solver(theta[inds], vr[inds])
            residual = abs(vr - vs_pre * np.cos(alpha_pre - theta))
            mask = np.array(residual) < self.threshold * vs_pre
            nbr_inlier = np.sum(mask)
            if nbr_inlier > max_nbr_inliers:
                max_nbr_inliers = nbr_inlier
                best_alpha_pre = alpha_pre
                best_vs_pre = vs_pre
                best_mask = mask
        return best_mask, best_alpha_pre, best_vs_pre

    def ransac_nusc(self, info, pcl, vis=False):
        v = (pcl[:, 6]**2 + pcl[:, 7]**2)**0.5                     # target's absolute velocity in radar frame
        v_comp = (pcl[:, 8]**2 + pcl[:, 9]**2)**0.5                                      # target's absolute velocity in global frame
        theta = np.arctan(pcl[:, 1] / (pcl[:, 0] + 1e-5))          # azimuth
        theta_v = np.arctan(pcl[:, 7] / (pcl[:, 6] + 1e-5))
        theta_r = theta - theta_v
        v_r = v * np.cos(theta_r)                                  # doppler range velocity
        low_sp_r = np.sum(np.abs(v_r) < 1) / v_r.shape[0]
        if low_sp_r > 0.5:
            best_mask = v_r < 1.0
            best_alpha_pre = None
            best_vs_pre = None
            # vis = True
        else:
            best_mask, best_alpha_pre, best_vs_pre = self.ransac(theta, v_r)
        o_r1 = 1 - np.sum(best_mask) / pcl.shape[0]
        o_r2 = np.sum(np.abs(v_comp) > 1) / v_comp.shape[0]
        if o_r1 > 0.25 or best_mask.shape[0] < 40:
            # vis = True
            pass

        if vis:
            print('vis scene:{}, frame:{}, sensor:{}, nbr:{}, o_r1:{}, o_r2:{}, low_sp_r:{}'.format(info[0][0], info[0][1], info[1], pcl.shape[0], o_r1, o_r2, low_sp_r))
            # --------------------------------------------- plot3 -------------------------------------------- #
            plt.rcParams['figure.figsize'] = (10, 10)
            plt.rcParams['figure.dpi'] = 150
            fig, ax = plt.subplots(2, 2)
            ax[1][0].scatter(np.rad2deg(theta), v_r, s=4, label='range velocity')
            ax[1][0].grid('on', lw=0.2)
            ax[1][0].set_xlim([-90, 90])
            ax[1][0].set_ylim([0, 30])
            ax[1][0].set_title('o_r1:{}/{}={:.2f} low_sp_r:{:.2f}'.format(pcl.shape[0] - np.sum(best_mask), pcl.shape[0], 1 - np.sum(best_mask) / pcl.shape[0], low_sp_r))
            ax[1][0].set_xlabel('azimuth(degree)')
            ax[1][0].set_ylabel('range velocity(m/s)')
            if low_sp_r > 0.5:
                ax[1][0].plot([np.min(np.rad2deg(theta)), np.max(np.rad2deg(theta))], [1, 1], c='green', lw=0.5)
                for ind in range(pcl.shape[0]):
                    if abs(v_r[ind]) > 1:
                        ax[1][0].scatter(np.rad2deg(theta[ind]), v_r[ind], s=1, c='r')
            else:
                theta_ = np.arange(-np.pi / 2, np.pi / 2, 0.1)
                ax[1][0].plot(np.rad2deg(theta_), best_vs_pre * np.cos(best_alpha_pre - theta_), label='estimated profile', color='green', lw=0.5)
                ax[1][0].fill_between(np.rad2deg(theta_),
                                      best_vs_pre * np.cos(best_alpha_pre - theta_) + self.threshold * best_vs_pre,
                                      best_vs_pre * np.cos(best_alpha_pre - theta_) - self.threshold * best_vs_pre,
                                      color='green',
                                      alpha=0.15)
            # --------------------------------------------- plot4 -------------------------------------------- #
            ax[1][1].scatter(np.rad2deg(theta), v, s=4, label='v = sqrt(vx^2+vy^2)')
            ax[1][1].scatter(np.rad2deg(theta), v_comp, s=4, label='v_comp(compensated by ego-motion)')
            ax[1][1].grid('on', lw=0.2)
            ax[1][1].set_xlim([-90, 90])
            ax[1][1].set_ylim([0, 15])
            ax[1][1].set_title('o_r2:{}/{}={:.2f}'.format(np.sum(np.abs(v_comp) > 1), v_comp.shape[0], np.sum(np.abs(v_comp) > 1) / v_comp.shape[0]))
            ax[1][1].set_xlabel('azimuth(degree)')
            ax[1][1].set_ylabel('v(m/s)')
            # ax[1][1].legend(loc=2)
            ax[1][1].plot([np.min(np.rad2deg(theta)), np.max(np.rad2deg(theta))], [1, 1], c='green', lw=0.5)
            for ind in range(pcl.shape[0]):
                if abs(v_comp[ind]) > 1:
                    ax[1][1].scatter(np.rad2deg(theta[ind]), v_comp[ind], s=1, c='r')

            # --------------------------------------------- plot1 -------------------------------------------- #
            X_LIMIT = [0, 100]
            Y_LIMIT = [-50, 50]
            v_amp = 1.0                                                                    # to make velocity arrow more visbile
            ax[0][0].scatter(pcl[:, 0], pcl[:, 1], s=1)
            for ind in range(pcl.shape[0]):
                ax[0][0].arrow(pcl[ind, 0], pcl[ind, 1], v_amp * pcl[ind, 6], v_amp * pcl[ind, 7], linewidth=0.5, head_width=1, fc='black', ec='r')

            ax[0][0].axis("square")
            # ax[0][0].set_xlim(X_LIMIT)
            # ax[0][0].set_ylim(Y_LIMIT)
            ax[0][0].grid('on', lw=0.2)
            ax[0][0].set_title('v = sqrt(vx^2+vy^2)')
            ax[0][0].set_xlabel('x(m)')
            ax[0][0].set_ylabel('y(m)')

            # --------------------------------------------- plot2 -------------------------------------------- #
            ax[0][1].scatter(pcl[:, 0], pcl[:, 1], s=1)
            for ind in range(pcl.shape[0]):
                ax[0][1].arrow(pcl[ind, 0], pcl[ind, 1], v_amp * pcl[ind, 8], v_amp * pcl[ind, 9], linewidth=0.5, head_width=1, fc='black', ec='r')
            ax[0][1].axis("square")
            # ax[0][1].set_xlim(X_LIMIT)
            # ax[0][1].set_ylim(Y_LIMIT)
            ax[0][1].grid('on', lw=0.2)
            ax[0][1].set_title('v_comp(compensated by ego-motion)')
            ax[0][1].set_xlabel('x(m)')
            ax[0][1].set_ylabel('y(m)')

            output_file = os.path.join('tmp/warning', 'scene{}_frame{}_sensor_{}_count{}.jpg'.format(info[0][0], info[0][1], info[1], info[2]))
            plt.savefig(output_file)
            plt.close()

            # visulize DTR on count4
            if info[2] == 0:
                import pickle
                with open(os.path.join('vis_DTR_cache/vr.pickle'), 'wb') as handle:
                    feature = {'theta': theta, 'v_r': v_r, 'best_alpha_pre': best_alpha_pre, 'best_vs_pre':best_vs_pre}
                    pickle.dump(feature, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return best_mask, best_alpha_pre, best_vs_pre
