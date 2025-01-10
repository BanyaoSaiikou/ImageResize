
import numpy as np
from math import cos, sin, pi
import cv2
import random

from py_rmpe_server.py_rmpe_config import RmpeGlobalConfig, TransformationParams


class AugmentSelection:
    def __init__(self, flip=False, degree=0., crop=(0, 0), scale=1.):
        self.flip = flip
        self.degree = degree  # rotate
        self.crop = crop  # shift actually
        self.scale = scale

    @staticmethod
    def random():
        flip = random.uniform(0., 1.) > TransformationParams.flip_prob
        degree = random.uniform(-1., 1.) * \
            TransformationParams.max_rotate_degree
        # TODO: see 'scale improbability' TODO above
        if random.uniform(0., 1.) > TransformationParams.scale_prob:
            scale_min = TransformationParams.scale_min
            scale_max = TransformationParams.scale_max
            scale = (scale_max - scale_min)*random.uniform(0., 1.) + scale_min
        else:
            scale = 1

        center_perterb_max = TransformationParams.center_perterb_max
        x_offset = int(random.uniform(-1., 1.) * center_perterb_max)
        y_offset = int(random.uniform(-1., 1.) * center_perterb_max)

        return AugmentSelection(flip, degree, (x_offset, y_offset), scale)

    @staticmethod
    def unrandom():
        flip = False
        degree = 0.
        scale = 1.
        x_offset = 0
        y_offset = 0

        return AugmentSelection(flip, degree, (x_offset, y_offset), scale)

    def affine(self, center, scale_self):

        # The main idea:
        # We will do all image transformations with one affine matrix.
        # This saves lot of cpu and make code significantly shorter.
        # Same affine matrix could be used
        # to transform joint coordinates afterwards

        A = self.scale * cos(self.degree / 180. * pi)
        B = self.scale * sin(self.degree / 180. * pi)

        scale_size = TransformationParams.target_dist / scale_self * self.scale

        (width, height) = center
        center_x = width + self.crop[0]
        center_y = height + self.crop[1]

        center2zero = np.array([[1., 0., -center_x],
                                [0., 1., -center_y],
                                [0., 0., 1.]])

        rotate = np.array([[A, B, 0],
                           [-B, A, 0],
                           [0, 0, 1.]])

        scale = np.array([[scale_size, 0, 0],
                          [0, scale_size, 0],
                          [0, 0, 1.]])

        flip = np.array([[-1 if self.flip else 1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])

        center2center = np.array([[1., 0., RmpeGlobalConfig.width//2],
                                  [0., 1., RmpeGlobalConfig.height//2],
                                  [0., 0., 1.]])

        # Order of combination is reversed
        combined = center2center.dot(flip).dot(scale)
        combined = combined.dot(rotate).dot(center2zero)

        return combined[0:2]


class Transformer:
    @staticmethod
    def transform(img, mask, meta, aug=AugmentSelection.random()):

        # Warp picture and mask
        M = aug.affine(meta['objpos'][0], meta['scale_provided'][0])

        # TODO: Need to understand this, scale_provided[0] is height of
        # main person divided by 368, caclulated in generate_hdf5.py
        # print(img.shape)
        img = cv2.warpAffine(img, M,
                             (RmpeGlobalConfig.height, RmpeGlobalConfig.width),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(127, 127, 127))
        mask = cv2.warpAffine(mask, M,
                              (RmpeGlobalConfig.height, RmpeGlobalConfig.width),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=255)
        # TODO: should be combined with warp for speed
        mask = cv2.resize(mask, RmpeGlobalConfig.mask_shape,
                          interpolation=cv2.INTER_CUBIC)
        # _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        # assert np.all((mask == 0) | (mask == 255)), \
        #     "Interpolation of mask should be thresholded only 0 or 255\n"
        mask = mask.astype(np.float) / 255.

        # Warp key points
        # TODO: Joint could be cropped by augmentation,
        # in this case we should mark it as invisible.
        # Update: May be we don't need it actually,
        # original code removed part sliced more than half totally,
        # may be we should keep it.
        original_points = meta['joints'].copy()
        # We reuse 3rd column in completely different way here, it is hack
        original_points[:, :, 2] = 1
        converted_points = np.matmul(M, original_points.transpose([0, 2, 1]))
        converted_points = converted_points.transpose([0, 2, 1])
        meta['joints'][:, :, 0:2] = converted_points

        # We just made image flip, i.e. right leg just became left leg,
        # and vice versa.

        # if aug.flip:
        #     tmpLeft = meta['joints'][:, RmpeGlobalConfig.leftParts, :]
        #     tmpRight = meta['joints'][:, RmpeGlobalConfig.rightParts, :]
        #     meta['joints'][:, RmpeGlobalConfig.leftParts, :] = tmpRight
        #     meta['joints'][:, RmpeGlobalConfig.rightParts, :] = tmpLeft

        # print(img.shape)

        return img, mask, meta
