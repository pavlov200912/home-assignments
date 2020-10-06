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
        self.shi_params = dict(maxCorners=1500, qualityLevel=0.02, minDistance=3, blockSize=7)
        self.lucas_params = dict(winSize=(15, 15), maxLevel=2,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.IQR_const = 3
        self.backtrack_threshold = 0.3

    @staticmethod
    def get_level_feature_size(level):
        """
        returns radius for feature at the pyramid's level
        level=0: original image -> smallest features
        level=depth: smallest image -> biggest features
        """
        # I recommend to use linear dependence: 5 * (level + 2)
        # But in task, it's specified to use "proportional" to the image size dependency
        return 6 * (1 << level)

    def pyramidal_detection(self, img, mask=None, depth=2, is_initial_frame=False):
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
            level_features = self.detect_features(level_image, level_mask, is_initial_frame) * (1 << level)
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

        for point, size in zip(features.reshape(-1, 2), feature_sizes):
            mask = cv2.circle(mask,
                              tuple(np.array(point).astype(int)),
                              int(size),
                              thickness=-1,
                              color=0)
        return mask

    def detect_features(self, image, mask, is_initial_frame=False):
        """
        :param is_initial_frame: In case of initial frame, we should lower features quality
        :param image: grayscale image
        :param mask: mask for shi-algorithm, mask[i][j] in {0, 255}
        :return: features to track in shape(-1, 2)
        """
        if is_initial_frame:
            self.shi_params['qualityLevel'] /= 2

        features = cv2.goodFeaturesToTrack((image * 255).astype(np.uint8), mask=mask, **self.shi_params)

        if is_initial_frame:
            self.shi_params['qualityLevel'] *= 2

        return features.reshape(-1, 2) if features is not None else np.array([]).reshape(-1, 2)

    def get_sparse_optical_flow(self, prev_image, cur_image, features):
        """
        Calculate optical flow for features & filter features with large error
        """
        new_features, status, error = cv2.calcOpticalFlowPyrLK(
            (prev_image * 255).astype(np.uint8),
            (cur_image * 255).astype(np.uint8),
            features.astype('float32').reshape((-1, 1, 2)),
            None,
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            **self.lucas_params
        )

        # Using backtracking for filtering features performs better, than error thresholds (subjectively)
        back_features, back_status, back_error = cv2.calcOpticalFlowPyrLK(
            (cur_image * 255).astype(np.uint8),
            (prev_image * 255).astype(np.uint8),
            new_features.astype('float32').reshape((-1, 1, 2)),
            None,
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            **self.lucas_params
        )

        forward_back_distance = abs(features.reshape(-1, 1, 2) - back_features).reshape(-1, 2).max(-1)
        back_tracked = forward_back_distance < self.backtrack_threshold

        filter_status = status == 1

        filter_mask = (filter_status & back_tracked.reshape(-1, 1)).reshape(-1)

        new_features = new_features if new_features is not None else np.array([]).reshape(-1, 2)
        return new_features, filter_mask

    def track_features(self, frame_sequence: pims.FramesSequence,
                       builder: _CornerStorageBuilder):
        image_0 = frame_sequence[0]

        features, sizes = self.pyramidal_detection(image_0, is_initial_frame=True)
        ids = np.array(list(range(0, len(features))))
        corners = FrameCorners(
            ids,
            features,
            sizes
        )
        max_corner_id = len(features)
        builder.set_corners_at_frame(0, corners)

        for frame, image_1 in enumerate(frame_sequence[1:], 1):
            mask = self.get_feature_mask(features, sizes, image_1.shape)

            tracked_features, error_mask = self.get_sparse_optical_flow(image_0, image_1, features)
            ids = ids[error_mask]
            tracked_features = tracked_features[error_mask]
            sizes = sizes[error_mask]

            new_features, new_sizes = self.pyramidal_detection(image_1, mask)
            new_ids = np.arange(max_corner_id, max_corner_id + len(new_features))
            max_corner_id = max_corner_id + len(new_features)

            ids = np.concatenate((ids, new_ids))
            tracked_features = np.concatenate((tracked_features, new_features.reshape(-1, 1, 2)))
            sizes = np.concatenate((sizes, new_sizes))

            corners = FrameCorners(
                ids,
                tracked_features,
                sizes
            )
            builder.set_corners_at_frame(frame, corners)
            image_0 = image_1
            features = tracked_features


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    detector = FeatureDetector()
    detector.track_features(frame_sequence, builder)


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
