import numpy as np
import cv2
from math import sqrt, isnan

from py_rmpe_server.py_rmpe_config import RmpeGlobalConfig, TransformationParams


class Heatmapper:

    def __init__(self, sigma=TransformationParams.sigma,
                 thre=TransformationParams.paf_thre):

        self.double_sigma2 = 2 * sigma * sigma
        self.thre = thre

        # Cached common parameters which same for all iterations and pictures
        stride = RmpeGlobalConfig.stride
        width = RmpeGlobalConfig.width//stride
        height = RmpeGlobalConfig.height//stride

        # This is coordinates of centers of bigger grid
        self.grid_x = np.arange(width)*stride + stride/2-0.5
        self.grid_y = np.arange(height)*stride + stride/2-0.5

        self.Y, self.X = np.mgrid[0:RmpeGlobalConfig.height:stride,
                                  0:RmpeGlobalConfig.width:stride]

        # TODO: Check it again.
        # Basically we should use center of grid,
        # but in this place classic implementation uses left-top point.
        # self.X = self.X + stride / 2 - 0.5
        # self.Y = self.Y + stride / 2 - 0.5

    def create_heatmaps(self, joints, img, mask):
        heatmaps = np.zeros(RmpeGlobalConfig.parts_shape, dtype=np.float)

        self.put_joints(heatmaps, joints)
        sl = slice(RmpeGlobalConfig.heat_start,
                   RmpeGlobalConfig.heat_start + RmpeGlobalConfig.heat_layers)
        heatmaps[RmpeGlobalConfig.bkg_start] = 1. - np.amax(heatmaps[sl, :, :],
                                                            axis=0)

        # self.put_limbs(heatmaps, joints)
        # heatmaps *= mask
        heatmaps_img = img.copy()
        heatmaps_img = cv2.resize(heatmaps_img, RmpeGlobalConfig.mask_shape)
        heatmaps_img[:, :, 0] += (heatmaps[0, :, :] * 255).astype(np.uint8)

        return heatmaps, heatmaps_img

    def put_gaussian_maps(self, heatmaps, layer, joints):

        # Actually exp(a+b) = exp(a)*exp(b),
        # lets use it calculating 2d exponent, it could just be calculated by
        for i in range(joints.shape[0]):
            exp_x = np.exp(-(self.grid_x-joints[i, 0])**2/self.double_sigma2)
            exp_y = np.exp(-(self.grid_y-joints[i, 1])**2/self.double_sigma2)

            exp = np.outer(exp_y, exp_x)

            # Note this is correct way of combination - min(sum(...),1.0)
            # as was in C++ code is incorrect
            # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/118
            heatmaps[RmpeGlobalConfig.heat_start + layer, :, :] = \
                np.maximum(heatmaps[RmpeGlobalConfig.heat_start + layer, :, :],
                           exp)

    def put_joints(self, heatmaps, joints):
        for i in range(RmpeGlobalConfig.num_parts):
            visible = joints[:, i, 2] < 2
            self.put_gaussian_maps(heatmaps, i, joints[visible, i, 0:2])
            # print(joints[visible, i, 0:2])

    def put_vector_maps(self, heatmaps, layerX, layerY, joint_from, joint_to):

        count = np.zeros(heatmaps.shape[1:], dtype=np.int)

        for i in range(joint_from.shape[0]):
            (x1, y1) = joint_from[i]
            (x2, y2) = joint_to[i]

            dx = x2-x1
            dy = y2-y1
            dnorm = sqrt(dx*dx + dy*dy)

            if dnorm == 0:  # We get nan here sometimes, it's kills NN
                # TODO: Handle it better.
                # Probably we should add zero paf, centered paf,
                # or skip this completely
                print("Parts are too close to others. Length is zero. Skipping")
                continue

            dx = dx / dnorm
            dy = dy / dnorm

            assert not isnan(dx) and not isnan(dy), "dnorm is zero, wtf"

            min_sx, max_sx = (x1, x2) if x1 < x2 else (x2, x1)
            min_sy, max_sy = (y1, y2) if y1 < y2 else (y2, y1)

            min_sx = int(round((min_sx - self.thre) / RmpeGlobalConfig.stride))
            min_sy = int(round((min_sy - self.thre) / RmpeGlobalConfig.stride))
            max_sx = int(round((max_sx + self.thre) / RmpeGlobalConfig.stride))
            max_sy = int(round((max_sy + self.thre) / RmpeGlobalConfig.stride))

            # Check PAF off screen.
            # Do not really need to do it with max > grid size
            if max_sy < 0:
                continue

            if max_sx < 0:
                continue

            if min_sx < 0:
                min_sx = 0

            if min_sy < 0:
                min_sy = 0

            # TODO: Check it again
            # This mask is not only speed up but crops paf really.
            # This copied from original code
            slice_x = slice(min_sx, max_sx)  # +1
            # int g_y = min_y; g_y < max_y; g_y++ -- note strict <
            slice_y = slice(min_sy, max_sy)  # +1

            dist = distances(self.X[slice_y, slice_x],
                             self.Y[slice_y, slice_x], x1, y1, x2, y2)
            dist = dist <= self.thre

            # TODO: Averaging by pafs mentioned in the paper
            # but never worked in C++ augmentation code.
            # += dist * dx
            heatmaps[layer_x, slice_y, slice_x][dist] = (dist * dx)[dist]
            # += dist * dy
            heatmaps[layer_y, slice_y, slice_x][dist] = (dist * dy)[dist]
            count[slice_y, slice_x][dist] += 1

        # TODO: averaging by pafs mentioned in the paper
        # but never worked in C++ augmentation code
        # heatmaps[layer_x, :, :][count > 0] /= count[count > 0]
        # heatmaps[layer_y, :, :][count > 0] /= count[count > 0]

    def put_limbs(self, heatmaps, joints):

        for (i, (fr, to)) in enumerate(RmpeGlobalConfig.limbs_conn):

            visible_from = joints[:, fr, 2] < 2
            visible_to = joints[:, to, 2] < 2
            visible = visible_from & visible_to

            layerX, layerY = (RmpeGlobalConfig.paf_start + i*2,
                              RmpeGlobalConfig.paf_start + i*2 + 1)
            self.put_vector_maps(heatmaps, layerX, layerY,
                                 joints[visible, fr, 0:2],
                                 joints[visible, to, 0:2])


# Parallel calculation distance from any points number of arbitrary shape(X, Y),
# to line defined by segment (x1,y1) -> (x2, y2)
def distances(X, Y, x1, y1, x2, y2):

    # Classic formula is:
    # d = (x2-x1)*(y1-y)-(x1-x)*(y2-y1)/sqrt((x2-x1)**2 + (y2-y1)**2)

    xD = (x2-x1)
    yD = (y2-y1)
    norm2 = sqrt(xD**2 + yD**2)
    dist = xD*(y1-Y)-(x1-X)*yD
    dist /= norm2

    return np.abs(dist)


def test():
    hm = Heatmapper()
    d = distances(hm.X, hm.Y, 100, 100, 50, 150)
    print(d < 8.)


if __name__ == "__main__":
    np.set_printoptions(precision=1, linewidth=1000,
                        suppress=True, threshold=100000)
    test()
