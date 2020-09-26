#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
from scipy.interpolate import interp2d

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class FeatureDetector:
    def __init__(self):
        self.shi_params = dict(maxCorners=1000, qualityLevel=0.075, minDistance=3, blockSize=7)
        self.lucas_params = dict(winSize=(15, 15), maxLevel=3,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.quality_decrease = {'manual': 2, 'cv2': 10}

    @staticmethod
    def get_level_feature_size(level):
        return 5 * (level + 2)

    def calculate_eigens(self, image, is_cv2=True):
        """
        calculate min eigenvalues for image, cv2 and manual variants
        cv2 variant is much faster, but values are differ from manual way (don't know why)
        manual way get higher
        """
        # If you use manual way, recommended to change constant in get_point_score method
        if is_cv2:
            return cv2.cornerMinEigenVal(image, self.shi_params['blockSize'])
        else:
            dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, 3)
            dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, 3)
            dx2 = dx ** 2
            dy2 = dy ** 2
            dxy = dx * dy
            CORNER_SIZE = tuple([self.shi_params['blockSize']] * 2)
            dx2_sum = cv2.GaussianBlur(dx2, CORNER_SIZE, 0)
            dy2_sum = cv2.GaussianBlur(dy2, CORNER_SIZE, 0)
            dxy_sum = cv2.GaussianBlur(dxy, CORNER_SIZE, 0)
            mats = np.dstack((dx2_sum, dxy_sum, dxy_sum, dy2_sum))
            h, w = image.shape
            mats = mats.reshape((h, w, 2, 2))
            return np.linalg.eigvals(mats).min(axis=2)

    @staticmethod
    def find_quality(eigvals_interpt2d, corner):
        return eigvals_interpt2d(corner[0], corner[1])

    @staticmethod
    def get_point_score(point, image_eigens):
        # TODO: should I replace local maximum with interpolation?
        x, y = int(point[0]), int(point[1])
        h, w = image_eigens.shape
        vx = [-3, -2, -1, 0, 1, 2, 3]
        vy = [-3, -2, -1, 0, 1, 2, 3]
        local_maxima = -1
        for dx in vx:
            for dy in vy:
                first = max(0, min(h, y + dy))
                second = max(0, min(w, x + dx))
                local_maxima = max(local_maxima, image_eigens[first][second])
        return local_maxima

    def get_features_quality_mask(self, image, features, eigens, quality_decrease, quality_level=None):
        # Without quality_decrease constant, you almost always have zero features with enough quality
        if quality_level is None:
            quality_level = self.shi_params['qualityLevel'] / quality_decrease
        mask = np.array([self.get_point_score(point, eigens) >= quality_level for point in features], dtype=bool)
        return mask

    def pyramidal_detection(self, img, mask=None, depth=3):
        """
        Construct depth layers of pyramid, calculate features in each layer.
        Features are filtered, using masks.
        Features from lower levels (bigger image) are considered more valuable
        """
        level_image = img.copy()
        features = np.array([], dtype=float).reshape(-1, 2)
        sizes = np.array([], dtype=float).reshape(-1, 1)
        if mask is not None:
            level_mask = mask.copy()
        else:
            level_mask = self.get_feature_mask(features, sizes, img.shape)
        for level in range(depth):
            level_features = self.detect_features(level_image, level_mask) * (1 << level)
            feature_size = self.get_level_feature_size(level)
            features = np.concatenate((features, level_features))
            sizes = np.concatenate((sizes,
                                    np.full((level_features.shape[0], 1), feature_size)))
            level_mask = self.get_feature_mask(features, sizes, level_image.shape, level_mask)
            level_image = cv2.pyrDown(level_image)
            level_mask = level_mask[::2, ::2]
        return features, sizes

    @staticmethod
    def get_feature_mask(features, feature_sizes, img_shape, mask=None):
        """
        Creates mask for goodFeaturesToTrack function, values are {0, 255}
        0 - can't access this image point
        255 - image point is available
        Bu default mask = [255]^(image.shape)
        """
        if mask is None:
            mask = np.full(img_shape, 255).astype(np.uint8)

        # mask always has type np.array, but openCV fails without this line...
        # Checkout https://github.com/opencv/opencv/issues/18120
        mask = np.array(mask)

        for point, size in zip(features, feature_sizes):
            mask = cv2.circle(mask,
                              tuple(np.array(point).astype(int)),
                              int(size),
                              0)
        return mask

    def detect_features(self, image, mask):
        """
        :param image: grayscale image
        :param mask: mask for shi-algorithm, mask[i][j] in {0, 255}
        :return: features to track in shape(-1, 2)
        """
        features = cv2.goodFeaturesToTrack((image * 255).astype(np.uint8), mask=mask, **self.shi_params)
        return features.reshape(-1, 2) if features is not None else np.array([]).reshape(-1, 2)

    def get_sparse_optical_flow(self, prev_image, cur_image, features):
        """
        :param prev_image, cur_image: grayscale images
        """
        new_features, status, error = cv2.calcOpticalFlowPyrLK(
            (prev_image * 255).astype(np.uint8),
            (cur_image * 255).astype(np.uint8),
            features.astype('float32').reshape((-1, 1, 2)),
            None,
            **self.lucas_params
        )
        # TODO: Use error value
        return new_features[status == 1].reshape(-1, 2) if new_features is not None \
            else np.array([]).reshape(-1, 2)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    detector = FeatureDetector()
    features, sizes = detector.pyramidal_detection(image_0)
    corners = FrameCorners(
        np.array(list(range(0, len(features)))),
        features,
        sizes
    )
    builder.set_corners_at_frame(0, corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        mask = detector.get_feature_mask(features, sizes, image_1.shape)

        next_features = detector.get_sparse_optical_flow(image_0, image_1, features)

        eigenvalues = detector.calculate_eigens(image_1, is_cv2=False)
        quality_mask = detector.get_features_quality_mask(image_1, next_features, eigenvalues,
                                                          quality_decrease=detector.quality_decrease['cv2'])
        next_features = next_features[quality_mask.reshape(-1)]

        if len(next_features) == 0:
            next_features = features

        corners = FrameCorners(
            np.array(list(range(0, len(next_features)))),
            next_features,
            np.array([10] * len(next_features))
        )
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1
        features = next_features


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
