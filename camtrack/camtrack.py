#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from collections import defaultdict
from typing import List, Optional, Tuple
import time

import numpy as np

import cv2
import sortednp as snp

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    TriangulationParameters,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    compute_reprojection_errors,
    Correspondences
)

from _corners import without_short_tracks


def pnp_new_frame(unprocessed_frames, points3d, ids, corner_storage, intrinsic_mat):
    max_inliers_len = -1
    new_frame, new_pos = None, None
    for frame in unprocessed_frames:
        intersection_ids, (idx_1, idx_2) = snp.intersect(ids,
                                          corner_storage[frame].ids.flatten().astype(np.int32),
                                          indices=True)
        if len(idx_1) < RANSAC_POINTS_THRESHOLD:
            continue

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points3d[intersection_ids],
            imagePoints=corner_storage[frame].points[idx_2],
            cameraMatrix=intrinsic_mat,
            distCoeffs=None,
            reprojectionError=RANSAC_REPROJECTION_ERROR,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not retval or len(inliers) == 0:
            print('Ransac failed =( for frame', frame)
            continue

        if len(inliers) > max_inliers_len:

            retval, rvec, tvec = cv2.solvePnP(
                objectPoints=(points3d[intersection_ids])[inliers],
                imagePoints=(corner_storage[frame].points[idx_2])[inliers],
                cameraMatrix=intrinsic_mat,
                distCoeffs=None,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not retval:
                return None
            max_inliers_len = len(inliers)
            new_frame = frame
            new_pos = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

    return new_frame, new_pos, max_inliers_len


RANSAC_POINTS_THRESHOLD = 5
RANSAC_REPROJECTION_ERROR = 3
RANSAC_ITERATIONS = 100
RETRIANGULATION_ITER_LIMIT = 50
RETRIANGULATION_SIZE_LIMIT = 300
RETRIANGULATION_FRAME_COUNT_LIMIT = 25
RETRIANGULATION_RANSAC_ITERATIONS = 3
RETRIANGULATION_RANSAC_REPROJECTION_ERROR = 1.5
SHORT_THRESHOLD = 1


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    corner_storage = without_short_tracks(corner_storage, SHORT_THRESHOLD)

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    triangulation_params = TriangulationParameters(
        max_reprojection_error=2,
        min_triangulation_angle_deg=1,
        min_depth=0.1
    )

    # TODO: implement
    frame_count = len(corner_storage)
    view_mats = {}
    processed_frames = np.zeros(frame_count, dtype=np.bool)
    points3d = np.zeros((corner_storage.max_corner_id() + 1, 3), dtype=np.float64)
    points3d_scores = np.zeros(corner_storage.max_corner_id() + 1)
    cloud_ids = np.zeros(corner_storage.max_corner_id() + 1)


    frame1, pose1 = known_view_1
    frame2, pose2 = known_view_2
    view_mats[frame1] = pose_to_view_mat3x4(pose1)
    view_mats[frame2] = pose_to_view_mat3x4(pose2)

    corrs = build_correspondences(corner_storage[frame1],
                                  corner_storage[frame2])
    points = []
    while len(points) < 30:
        points, ids, median_cos = triangulate_correspondences(corrs,
                                                              pose_to_view_mat3x4(pose1),
                                                              pose_to_view_mat3x4(pose2),
                                                              intrinsic_mat,
                                                              triangulation_params)
        triangulation_params = TriangulationParameters(
            triangulation_params.max_reprojection_error * 2,
            triangulation_params.min_triangulation_angle_deg * 0.5,
            triangulation_params.min_depth
        )


    cloud_ids[ids] = 1
    points3d[ids] = points
    points3d_scores[ids] = 2

    print(f"Triangulation between initial frames done! Founded {len(points)} points with med_cos={np.degrees(np.arccos(median_cos))}")

    frames_for_corner = defaultdict(list)
    for frame in range(frame_count):
        frame_corners = corner_storage[frame]
        for index, feature_id in enumerate(frame_corners.ids.flatten()):
            frames_for_corner[feature_id].append((index, frame))

    iteration = 0
    last_retriangulate_time = {}

    while True:
        iteration += 1
        update = False
        unprocessed_frames = np.arange(len(corner_storage), dtype=np.int32)[~processed_frames]

        # Solve PnPRansac
        pnp_start = time.time()
        new_frame, new_pos, inliers_count = pnp_new_frame(unprocessed_frames,
                                                          points3d,
                                                          np.array(np.nonzero(cloud_ids)[0], dtype=np.int32),
                                                          corner_storage,
                                                          intrinsic_mat)
        pnp_end = time.time()
        if new_frame is not None:
            update = True
            processed_frames[new_frame] = True
            view_mats[new_frame] = new_pos


            """begin: RETRIANGULATION"""
            # Extract frames, choose posed one, limit them
            # Extract points to retriangulation, limit them
            # For each point Ransac frames with all frames as inliers
            # Choose best solution for this point
            #
            trian_start = time.time()
            point_to_retriangulate = []
            for id in corner_storage[new_frame].ids.flatten():
                if id not in last_retriangulate_time or iteration - last_retriangulate_time[id] > RETRIANGULATION_ITER_LIMIT:
                    point_to_retriangulate.append(id)

            if len(point_to_retriangulate) > RETRIANGULATION_SIZE_LIMIT:
                point_to_retriangulate = np.random.choice(point_to_retriangulate, RETRIANGULATION_SIZE_LIMIT, replace=False)


            def retriangulate_feature(corner):
                # Find corners on other frames corresponding to same corner_id
                frames_info = []
                for index, frame in frames_for_corner[corner]:
                    if frame in view_mats:
                        # Frame, 2d Point, View Matrix for Frame
                        frames_info.append((frame, corner_storage[frame].points[index], view_mats[frame]))

                if len(frames_info) < 2:
                     # Can't retriangulate
                    return None

                # Limit frames
                if len(frames_info) > RETRIANGULATION_FRAME_COUNT_LIMIT:
                    frame_ids = np.random.choice(np.arange(len(frames_info)), RETRIANGULATION_FRAME_COUNT_LIMIT, replace=False)
                    frames_info = np.array(frames_info, dtype=object)[frame_ids]
                else:
                    frames_info = np.array(frames_info, dtype=object)
                # RANSAC:
                new_point, point_inliers_score = -1, -1
                for _ in range(RETRIANGULATION_RANSAC_ITERATIONS):
                    id_frame1, id_frame2 = np.random.choice(len(frames_info), 2, replace=False)

                    frame1, p1, pos1 = frames_info[id_frame1]
                    frame2, p2, pos2 = frames_info[id_frame2]

                    p, _, _ = triangulate_correspondences(Correspondences(
                         np.array([0]), np.array(p1), np.array(p2)
                    ), pos1, pos2, intrinsic_mat, triangulation_params)

                    if len(p) == 0:
                        # triangulation_params can be too high, in this case we return None
                        return None

                    e = np.array(
                         [compute_reprojection_errors(p, point2d, intrinsic_mat @ mat) for frame, point2d, mat in frames_info]
                    )
                    good = e < RETRIANGULATION_RANSAC_REPROJECTION_ERROR
                    inliers_count = sum(good)[0]
                    if inliers_count > point_inliers_score:
                        point_inliers_score = inliers_count
                        new_point = p[0]

                    if len(frames_info) == 2:
                        # One iteration is enough in case of 2 frames
                        break

                return new_point, point_inliers_score if point_inliers_score > 0 else None


            updated_points = 0
            for point_id in point_to_retriangulate:
                last_retriangulate_time[point_id] = iteration
                result = retriangulate_feature(point_id)
                if result is None:
                    continue
                p, score = result
                if score is not None and score > points3d_scores[point_id]:
                    points3d_scores[point_id] = score
                    points3d[point_id] = p
                    cloud_ids[point_id] = True
                    updated_points += 1

            trian_end = time.time()
            print(f"{int(iteration/ frame_count * 100)}% "
                  f"## Iteration: {iteration} "
                  f"## Frame: {new_frame}"
                  f"## Cloud: {sum(cloud_ids)} "
                  f"## Retriangulated: {len(point_to_retriangulate)} "
                  f"## Updated: {updated_points} "
                  f"## PnpTime: {pnp_end - pnp_start} "
                  f"## TriangTime: {trian_end - trian_start}")

            """end: RETRIANGULATION"""


        if not update:
            print(f"Only: {sum(processed_frames) / frame_count} processed =(")
            break

    point_cloud_builder = PointCloudBuilder(np.array(np.nonzero(cloud_ids)[0], dtype=np.int32), points3d[np.array(cloud_ids).astype(bool)])

    view_mats = [view_mats[key] for key in sorted(view_mats.keys())]

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
