#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from collections import defaultdict
from typing import List, Optional, Tuple
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares

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
    Correspondences,
    eye3x4
)

from _corners import without_short_tracks


def pnp_new_frame(unprocessed_frames, points3d, ids, corner_storage, intrinsic_mat):
    max_inliers_len = -1
    new_frame, new_pos = None, None
    for frame in unprocessed_frames:
        intersection_ids, (idx_1, idx_2) = snp.intersect(ids,
                                                         corner_storage[frame].ids.flatten().astype(np.int32),
                                                         indices=True)
        if len(idx_1) < HyperParams.RANSAC_POINTS_THRESHOLD:
            continue

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points3d[intersection_ids],
            imagePoints=corner_storage[frame].points[idx_2],
            cameraMatrix=intrinsic_mat,
            distCoeffs=None,
            reprojectionError=HyperParams.RANSAC_REPROJECTION_ERROR,
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


class HyperParams:
    RANSAC_POINTS_THRESHOLD = 5
    RANSAC_REPROJECTION_ERROR = 3
    RANSAC_ITERATIONS = 100
    RETRIANGULATION_ITER_LIMIT = 50
    RETRIANGULATION_SIZE_LIMIT = 300
    RETRIANGULATION_FRAME_COUNT_LIMIT = 25
    RETRIANGULATION_RANSAC_ITERATIONS = 3
    RETRIANGULATION_RANSAC_REPROJECTION_ERROR = 1.5
    SHORT_THRESHOLD = 1  # Nice for usage when > 1, but ironman test bans this feature =((((

    @staticmethod
    def lower_thresholds():
        HyperParams.RANSAC_POINTS_THRESHOLD *= 2
        HyperParams.RANSAC_REPROJECTION_ERROR *= 2
        HyperParams.RETRIANGULATION_RANSAC_REPROJECTION_ERROR *= 2

    @staticmethod
    def lower_triangulation_params(triangulation_params):
        return TriangulationParameters(max_reprojection_error=triangulation_params.max_reprojection_error * 2,
                                       min_triangulation_angle_deg=triangulation_params.min_triangulation_angle_deg / 2,
                                       min_depth=0)


def calculate_initial_views(corner_storage, intrinsic_mat, triangulation_params):
    # Sort (filter) pairs of frames by
    # - features intersection
    # - frame distance
    min_intersection = 10
    top_pairs = []
    min_distance = 10
    max_distance = 25

    scaler = lambda x: x // 100 if x >= 1000 else x
    for frame1 in range(len(corner_storage)):
        for frame2 in range(frame1 + 1, len(corner_storage)):
            corrs = build_correspondences(corner_storage[frame1],
                                          corner_storage[frame2])
            intersect_len = len(corrs.ids)
            if intersect_len > min_intersection and min_distance <= frame2 - frame1 <= max_distance:
                top_pairs.append((frame1, frame2, intersect_len, frame2 - frame1))
    top_pairs = sorted(top_pairs, key=lambda x: (scaler(x[2]), x[3]), reverse=True)

    # print(*list(top_pairs)[:500], sep='\n')
    pairs_to_check = 500

    top_pairs_stats = []
    for frame1, frame2, _, _ in top_pairs[:pairs_to_check]:
        corrs = build_correspondences(corner_storage[frame1], corner_storage[frame2])
        assert len(corrs.ids) >= 5  # Otherwise, can't solve with cv2

        essential_mat, essential_inliers = cv2.findEssentialMat(corrs.points_1, corrs.points_2, intrinsic_mat,
                                                                method=cv2.RANSAC,
                                                                prob=0.995,
                                                                threshold=1.0)

        _, homography_inliers = cv2.findHomography(corrs.points_1, corrs.points_2,
                                                   method=cv2.RANSAC,
                                                   confidence=0.995,
                                                   ransacReprojThreshold=HyperParams.RANSAC_REPROJECTION_ERROR)

        ids_to_remove = np.where(essential_inliers == 0)[0]

        corrs = build_correspondences(corner_storage[frame1], corner_storage[frame2],
                                      ids_to_remove=ids_to_remove.astype(np.int32))

        _, R, t, triangulated_inliers = cv2.recoverPose(essential_mat, corrs.points_1, corrs.points_2, intrinsic_mat)

        # print(f"DEBUG: {frame1}:{frame2} corrs: {len(corrs.ids)} E_I:{np.count_nonzero(essential_inliers)} H_I: {np.count_nonzero(homography_inliers)} "
        #       f"R: {np.count_nonzero(essential_inliers) / np.count_nonzero(homography_inliers)} T_I:{np.count_nonzero(triangulated_inliers)}")

        points, ids, _ = triangulate_correspondences(corrs,
                                                     eye3x4(),
                                                     np.hstack((R, t)),
                                                     intrinsic_mat,
                                                     triangulation_params)

        triangulated_len = len(ids)
        # if scaler(triangulated_len) > best_pair_triangulated_len:
        #    result_frame_1, result_frame_2 = frame1, frame2
        #    best_pair_triangulated_len = scaler(triangulated_len)
        #    result_camera_pos = np.hstack((R, t))
        top_pairs_stats.append(((frame1, frame2),
                                triangulated_len,
                                np.count_nonzero(essential_inliers) / (np.count_nonzero(homography_inliers) + 1),
                                np.count_nonzero(triangulated_inliers),
                                np.count_nonzero(essential_inliers),
                                np.hstack((R, t))
                                ))

    top_pairs_stats = list(sorted(top_pairs_stats, key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True))
    best_pair_triangulated_len = top_pairs_stats[0][1]
    result_frame_1, result_frame_2 = top_pairs_stats[0][0]
    result_camera_pos = top_pairs_stats[0][-1]

    print(f"Best triangulated result: {best_pair_triangulated_len * 100}")
    return (result_frame_1, view_mat3x4_to_pose(eye3x4())), (result_frame_2, view_mat3x4_to_pose(result_camera_pos))


def bundle_adjustment(processed_frames, points3d, ids, view_mats, frames_for_corner, corner_storage, intrinsic_mat, bundle_adjustment_freq):
    t0 = time.time()
    # Indices of frames to BA

    min_intersection = 10
    top_frames = []

    ba_frames = np.arange(len(corner_storage), dtype=np.int32)[processed_frames]
    n_cameras = len(ba_frames)

    # Select frames with most intersection for bundle adjustment
    for frame_id1 in range(n_cameras):
        for frame_id2 in range(n_cameras):
            frame1, frame2 = ba_frames[frame_id1], ba_frames[frame_id2]
            corrs = build_correspondences(corner_storage[frame1],
                                          corner_storage[frame2])
            intersect_len = len(corrs.ids)
            if intersect_len > min_intersection:
                top_frames.append((frame1, bundle_adjustment_freq[frame1], intersect_len))
                top_frames.append((frame2, bundle_adjustment_freq[frame2], intersect_len))

    frames_bound = 25

    # TODO: Add frequency statistic for bundle adjustment of every frame
    top_frames = sorted(top_frames, key=lambda x: (-x[1], x[2]), reverse=True)
    ba_frames = set()
    for frame, _, _ in top_frames:
        if frame not in ba_frames:
            ba_frames.add(frame)
        if len(ba_frames) >= frames_bound:
            break
    ba_frames = np.array(list(ba_frames))
    n_cameras = len(ba_frames)

    # TODO: Choose only specific points for BA? (good, bad, with a lot of frames?)
    # TODO: Add frequency of bundle adjustment for every point
    ba_points_ids = ids
    n_points = len(ba_points_ids)

    print(f'Starting bundle adjustment with n_cameras: {n_cameras} and n_points: {n_points}')

    camera_params = []
    for frame in ba_frames:
        # Calculating camera params for frame (rotation vector, translation vector)
        view_mat = view_mats[frame]
        r_mat = view_mat[:, :3]  # (3, 3)
        t_vec = view_mat[:, 3]  # (3,)
        r_vec, _ = cv2.Rodrigues(r_mat)  # (3,1)
        p = np.hstack((r_vec.T, t_vec.reshape(-1, 1).T))  # (1, 6)
        camera_params.extend(p.flatten())
        bundle_adjustment_freq[ba_frames] += 1

    camera_params = np.array(camera_params)  # (6 * n_cameras, )

    ba_frames_for_corner = {}
    ba_camera_ids = []
    ba_points_3d_ids = []
    ba_points_2d = []
    for ba_id, feature_id in enumerate(ba_points_ids):
        feature_frames = np.array([x[1] for x in frames_for_corner[feature_id]])
        feature_id_for_frames = np.array([x[0] for x in frames_for_corner[feature_id]])
        intersection_ids, (idx1, idx2) = snp.intersect(feature_frames,
                                                       ba_frames,
                                                       indices=True)
        # idx2 - indices of frames for this feature in ba_frames array (with this id it's easy to find rot_vec for frame)
        ba_camera_ids.extend(list(idx2))

        ba_points_3d_ids.extend([ba_id] * len(idx2))

        # list of 2d points for this feature on frames
        ba_points_2d.extend([corner_storage[frame].points[index]
                             for index, frame in
                             zip(feature_id_for_frames[idx1], feature_frames[idx1])])



    ba_camera_ids = np.array(ba_camera_ids)
    ba_points_3d_ids = np.array(ba_points_3d_ids)
    ba_points_2d = np.array(ba_points_2d)

    print(f"Bundle Adjustment jacobian matrix size: {(n_cameras * 6 + n_points * 3, len(ba_camera_ids) * 2)}")

    ba_points_3d = points3d[ba_points_ids].flatten()  # 3 * n_points

    initial_p = np.hstack((camera_params, ba_points_3d))

    def rotate(points, rot_vecs):
        """Rotate points by given rotation vectors.

        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    def proj(p_3d, cam_params, intrinsic_mat):
        r_vecs = cam_params[:, :3]
        t_vecs = cam_params[:, 3:6]
        points_proj = rotate(p_3d, r_vecs)
        points_proj += t_vecs
        # Rotated, now we shall use camera params matrix
        # (3, 3) * (3, n)
        points_proj = np.dot(intrinsic_mat, points_proj.T).T
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        return points_proj

    def fun(params, n_cameras, n_points, cam_ids, p3d_ids, points_2d, intrinsic_mat):
        cam_params = params[: n_cameras * 6].reshape((n_cameras, 6))
        p_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = proj(p_3d[p3d_ids], cam_params[cam_ids], intrinsic_mat)
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

        return A

    A = bundle_adjustment_sparsity(n_cameras, n_points, ba_camera_ids, ba_points_3d_ids)

    res = least_squares(fun, initial_p, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, ba_camera_ids, ba_points_3d_ids, ba_points_2d, intrinsic_mat))

    p = np.array(res.x)
    camera_params = p[:n_cameras * 6].reshape((n_cameras, 6))
    p_3d = p[n_cameras * 6:].reshape((n_points, 3))

    for ba_frame_id, frame in enumerate(ba_frames):
        cam = camera_params[ba_frame_id]
        r_vec = cam[:3].reshape(-1, 1)
        t_vec = cam[3:].reshape(-1, 1)
        view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)

        view_mats[frame] = view_mat

    for ba_id, feature_id in enumerate(ba_points_ids):
        points3d[feature_id] = p_3d[ba_id]

    t1 = time.time()
    print("Bundle Adjustment finished, it took {0:.0f} seconds".format(t1 - t0))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    corner_storage = without_short_tracks(corner_storage, HyperParams.SHORT_THRESHOLD)

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

    if known_view_1 is None or known_view_2 is None:
        print("Finding best frames")
        known_view_1, known_view_2 = calculate_initial_views(corner_storage, intrinsic_mat, triangulation_params)
        print(f"Best frames found: {known_view_1[0]} and {known_view_2[0]}")

    frame_count = len(corner_storage)
    view_mats = {}
    processed_frames = np.zeros(frame_count, dtype=np.bool)
    points3d = np.zeros((corner_storage.max_corner_id() + 1, 3), dtype=np.float64)
    points3d_scores = np.zeros(corner_storage.max_corner_id() + 1)
    bundle_adjustment_freq = np.zeros(frame_count + 1)
    cloud_ids = np.zeros(corner_storage.max_corner_id() + 1)

    frame1, pose1 = known_view_1
    frame2, pose2 = known_view_2
    view_mats[frame1] = pose_to_view_mat3x4(pose1)
    view_mats[frame2] = pose_to_view_mat3x4(pose2)

    corrs = build_correspondences(corner_storage[frame1],
                                  corner_storage[frame2])
    points = []
    while len(points) < 10:
        print("Can't perform initial retriangulation. Updating params.")
        # print(len(points), triangulation_params)
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

    print(
        f"Triangulation between initial frames done! Founded {len(points)} points with med_cos={np.degrees(np.arccos(median_cos))}")

    frames_for_corner = defaultdict(list)
    for frame in range(frame_count):
        frame_corners = corner_storage[frame]
        for index, feature_id in enumerate(frame_corners.ids.flatten()):
            frames_for_corner[feature_id].append((index, frame))

    iteration = 0
    last_retriangulate_time = {}
    dooms_day_counter = 0
    while True:
        iteration += 1
        update = False
        unprocessed_frames = np.arange(len(corner_storage), dtype=np.int32)[~processed_frames]


        if len(unprocessed_frames) == 0:
            print("Done!")
            break

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
                if id not in last_retriangulate_time or iteration - last_retriangulate_time[
                    id] > HyperParams.RETRIANGULATION_ITER_LIMIT:
                    point_to_retriangulate.append(id)

            if len(point_to_retriangulate) > HyperParams.RETRIANGULATION_SIZE_LIMIT:
                point_to_retriangulate = np.random.choice(point_to_retriangulate,
                                                          HyperParams.RETRIANGULATION_SIZE_LIMIT, replace=False)

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
                if len(frames_info) > HyperParams.RETRIANGULATION_FRAME_COUNT_LIMIT:
                    frame_ids = np.random.choice(np.arange(len(frames_info)),
                                                 HyperParams.RETRIANGULATION_FRAME_COUNT_LIMIT, replace=False)
                    frames_info = np.array(frames_info, dtype=object)[frame_ids]
                else:
                    frames_info = np.array(frames_info, dtype=object)
                # RANSAC:
                new_point, point_inliers_score = -1, -1
                for _ in range(HyperParams.RETRIANGULATION_RANSAC_ITERATIONS):
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
                        [compute_reprojection_errors(p, point2d, intrinsic_mat @ mat) for frame, point2d, mat in
                         frames_info]
                    )
                    good = e < HyperParams.RETRIANGULATION_RANSAC_REPROJECTION_ERROR
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
            print(f"{int(iteration / frame_count * 100)}% "
                  f"## Iteration: {iteration} "
                  f"## Frame: {new_frame}"
                  f"## Cloud: {sum(cloud_ids)} "
                  f"## Retriangulated: {len(point_to_retriangulate)} "
                  f"## Updated: {updated_points} "
                  f"## PnpTime: {pnp_end - pnp_start} "
                  f"## TriangTime: {trian_end - trian_start}")

            """end: RETRIANGULATION"""
        #bundle_adjustment(processed_frames, points3d, np.array(np.nonzero(cloud_ids)[0], dtype=np.int32),
        #                  view_mats, frames_for_corner, corner_storage, intrinsic_mat, bundle_adjustment_freq)

        if not update:
            dooms_day_counter += 1
            print("CANT UPDATE ANYTHING, LOWING THRESHOLDS!")
            HyperParams.lower_thresholds()
            triangulation_params = HyperParams.lower_triangulation_params(triangulation_params)


            if dooms_day_counter >= 4:
                bundle_adjustment(processed_frames, points3d, np.array(np.nonzero(cloud_ids)[0], dtype=np.int32),
                                  view_mats, frames_for_corner, corner_storage, intrinsic_mat, bundle_adjustment_freq)

            if dooms_day_counter == 25:
                print(f"Failed to process all frames: {(1 - len(unprocessed_frames) / frame_count) * 100}% done")
                break
        else:
            dooms_day_counter = 0

    point_cloud_builder = PointCloudBuilder(np.array(np.nonzero(cloud_ids)[0], dtype=np.int32),
                                            points3d[np.array(cloud_ids).astype(bool)])

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
