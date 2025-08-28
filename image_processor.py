# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
# from latentsync.utils.util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Union
import face_alignment
from scipy.spatial import procrustes
from decord import VideoReader

def read_video(video_path: str):
    vr = VideoReader(video_path)
    video_frames = vr[:].asnumpy()
    vr.seek(0)
    return video_frames

def oob(
    face_detect_crop_coords,
    annotation_crop_coords,  # tuple, (tlx, tly, brx, bry)
) -> bool:
    tlx, tly, brx, bry = face_detect_crop_coords
    tlx_a, tly_a, brx_a, bry_a = annotation_crop_coords
    # for i, (tlx_a, tly_a, brx_a, bry_a) in enumerate(annotation_crop_coords):
    #     # If the annotation crop is out of the boundary of the face detect crop, return True
    #     if tlx_a < tlx or tly_a < tly or brx_a > brx or bry_a > bry:
    #         return True
    
    # if annotation_crop_coords[:, 0].min() < tlx or \
    #     annotation_crop_coords[:, 1].min() < tly or \
    #     annotation_crop_coords[:, 2].max() > brx or \
    #     annotation_crop_coords[:, 3].max() > bry:
    #     return True
    
    if tlx_a < tlx or tly_a < tly or brx_a > brx or bry_a > bry:
        return True
    return False

class FloatProcessor:
    """
    A class to process images and videos for face detection and cropping.
    This class uses the `face_alignment` library to detect faces in images and videos,
    and crops the detected faces to a specified resolution. It supports both single images
    and video files, resizing images as needed.

    1. Loads video and converts to (F, C, H, W).
    2. Uses the first frame to detect a face and define a crop window.
    3. Crops each frame consistently around that window.
    4. Pads and resizes the crop to a square of fixed resolution.
    5. Returns the entire video as a sequence of clean, aligned face crops.
    """
    def __init__(self, resolution=512):
        self.resolution = resolution
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=self.device
        )

    def filter_bboxes_by_size(self, bboxes, frame_shape, min_area_ratio=0.01, min_pixels=100 * 100):
        """
        Filter bounding boxes by minimum size constraints.

        Args:
            bboxes (list of tuples): Each bbox is (x1, y1, x2, y2, score)
            frame_shape (tuple): (H, W) of the frame
            min_area_ratio (float): Minimum area ratio to keep
            min_pixels (int): Minimum number of pixels in the bbox

        Returns:
            list of tuples: Filtered bounding boxes
        """
        filtered = []
        frame_area = frame_shape[0] * frame_shape[1]

        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            face_area = (x2 - x1) * (y2 - y1)
            if face_area >= min_pixels and (face_area / frame_area) >= min_area_ratio:
                filtered.append(bbox)

        return filtered


    @torch.no_grad()
    def process_img(self, img: np.ndarray, mult: float) -> tuple[int, int, int, int, float] | None:
        img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        if img is None or img.size == 0:
            return None

        resized_img = cv2.resize(
            img,
            dsize=(0, 0),
            fx=mult,
            fy=mult,
            interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC
        )

        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [
            (int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score)
            for (x1, y1, x2, y2, score) in bboxes
            if score > 0.95
        ]
        return bboxes[0] if bboxes else None


    def crop_video(self, video_path: str) -> np.ndarray:
        from einops import rearrange
        video_frames = read_video(video_path)
        video_frames = np.transpose(video_frames, (0, 3, 1, 2))  # (F, C, H, W)

        img = video_frames[0]
        if img is None or img.size == 0:
            raise RuntimeError("Failed to read the first frame")

        mult = 360. / img.shape[1]  # img.shape[1] is height in CHW
        bbox = self.process_img(img, mult)
        if bbox is None:
            raise RuntimeError("No face detected in the first frame")

        bsy = int((bbox[3] - bbox[1]) / 2)
        bsx = int((bbox[2] - bbox[0]) / 2)
        my = int((bbox[1] + bbox[3]) / 2)
        mx = int((bbox[0] + bbox[2]) / 2)
        bs = int(max(bsy, bsx) * 1.6)  # A square box size that includes some margin for more context like forehead, chin, ears.

        results = []
        for i, frame in enumerate(video_frames):
            frame = np.transpose(frame, (1, 2, 0)) # CHW → HWC
            frame = cv2.copyMakeBorder(
                frame,
                bs, bs, bs, bs,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )

            my_padded = my + bs
            mx_padded = mx + bs
            h, w = frame.shape[:2]

            crop_y1 = max(0, my_padded - bs)
            crop_y2 = min(h, my_padded + bs)
            crop_x1 = max(0, mx_padded - bs)
            crop_x2 = min(w, mx_padded + bs)

            crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop_img.size == 0:
                print(f"Invalid crop for frame {i}")
                continue

            crop_img = cv2.resize(
                crop_img,
                dsize=(self.resolution, self.resolution),
                interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC
            )

            crop_img = rearrange(torch.from_numpy(crop_img), "h w c -> c h w")
            results.append(crop_img)

        if not results:
            raise RuntimeError("No valid frames processed")

        results = torch.stack(results)
        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


    @torch.no_grad()
    def process_img_multiple_faces(self, img: np.ndarray, mult: float) -> list[tuple[int, int, int, int, float]]:
        img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        if img is None or img.size == 0:
            return []

        resized_img = cv2.resize(
            img,
            dsize=(0, 0),
            fx=mult,
            fy=mult,
            interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC
        )

        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [
            (int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score)
            for (x1, y1, x2, y2, score) in bboxes
            if score > 0.95
        ]
        return bboxes


    def crop_video_multiple_faces(
        self, 
        video_frames: np.array, 
        video_path: str | None = None,
        annotation_crop_coords: np.ndarray | None = None,
    ) -> dict[int, np.ndarray] | None:
        from einops import rearrange

        if video_frames is None:
            assert video_path is not None, "video_path should not be None if video_frames is None"
            video_frames = read_video(video_path)
        video_frames = np.transpose(video_frames, (0, 3, 1, 2))  # (F, C, H, W)

        img = video_frames[0]
        if img is None or img.size == 0:
            raise RuntimeError("Failed to read the first frame")

        mult = 360. / img.shape[1]
        bboxes = self.process_img_multiple_faces(img, mult)

        # Filter out small faces
        bboxes = self.filter_bboxes_by_size(bboxes, img.shape[:2], min_area_ratio=0.028)
        if not bboxes:
            raise RuntimeError("No sufficiently large faces detected in the first frame")

        all_face_results = {}

        for face_idx, bbox in enumerate(bboxes):
            # Center of the face bounding box
            bsy = int((bbox[3] - bbox[1]) / 2)
            bsx = int((bbox[2] - bbox[0]) / 2)
            my = int((bbox[1] + bbox[3]) / 2)
            mx = int((bbox[0] + bbox[2]) / 2)

            # Expand crop area to include head/neck/shoulders
            # expand_scale_y = 2.0
            # expand_scale_x = 1.5
            expand_scale_y = 1.6
            expand_scale_x = 1.6
            bs_y = int(bsy * expand_scale_y)
            bs_x = int(bsx * expand_scale_x)
            bs = max(bs_y, bs_x)  # ensure square crop

            results = []
            for i, frame in enumerate(video_frames):
                frame = np.transpose(frame, (1, 2, 0))  # CHW → HWC
                h, w = frame.shape[:2]

                # Compute crop box
                crop_y1 = max(0, my - bs)
                crop_y2 = min(h, my + bs)
                crop_x1 = max(0, mx - bs)
                crop_x2 = min(w, mx + bs)
                
                if annotation_crop_coords is not None:
                    if oob((crop_x1, crop_y1, crop_x2, crop_y2), annotation_crop_coords):
                        print(f"Face {face_idx}, frame {i} is out of the annotation crop")
                        return None

                crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop_img.size == 0:
                    print(f"Invalid crop for face {face_idx}, frame {i}")
                    continue

                # Make square by adding white padding if needed
                ch, cw = crop_img.shape[:2]
                diff = abs(ch - cw)
                pad_top = pad_bottom = pad_left = pad_right = 0
                if ch > cw:
                    pad_left = diff // 2
                    pad_right = diff - pad_left
                elif cw > ch:
                    pad_top = diff // 2
                    pad_bottom = diff - pad_top

                crop_img = cv2.copyMakeBorder(
                    crop_img,
                    pad_top, pad_bottom, pad_left, pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(255, 255, 255)  # white padding
                )

                # Resize to target resolution
                crop_img = cv2.resize(
                    crop_img,
                    dsize=(self.resolution, self.resolution),
                    interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC
                )

                crop_img = rearrange(torch.from_numpy(crop_img), "h w c -> c h w")
                results.append(crop_img)

            if results:
                stacked = torch.stack(results)
                all_face_results[face_idx] = rearrange(stacked, "f c h w -> f h w c").numpy()

        if not all_face_results:
            raise RuntimeError("No valid face crops were extracted")

        return all_face_results


    @torch.no_grad()
    def estimate_pose(self, image: np.ndarray) -> tuple[float, float, float] | None:
        landmarks = self.fa.get_landmarks(image)
        if landmarks is None or len(landmarks) == 0:
            return None

        image_points = np.array([
            landmarks[0][30],  # Nose tip
            landmarks[0][8],   # Chin
            landmarks[0][36],  # Left eye left corner
            landmarks[0][45],  # Right eye right corner
            landmarks[0][48],  # Left mouth corner
            landmarks[0][54]   # Right mouth corner
        ], dtype=np.float32)

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float32)

        height, width = image.shape[:2]
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        if not success:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        rotation_mat = np.asarray(rotation_mat, dtype=np.float32)
        pose_mat = cv2.hconcat([rotation_mat, np.array([[0], [0], [0]], dtype=np.float32)])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = euler_angles.flatten()
        return pitch, yaw, roll


    @torch.no_grad()
    def estimate_pose_from_video(self, video_path: str, yaw_thresh: float = 45.0, pitch_thresh: float = 45.0, max_invalid_ratio: float = 0.5) -> bool:
        video_frames = read_video(video_path)
        video_frames = np.transpose(video_frames, (0, 3, 1, 2))  # (F, C, H, W)
        total_frames = len(video_frames)
        invalid_count = 0

        for frame in video_frames[::10]:
            img = np.transpose(frame, (1, 2, 0))  # CHW → HWC
            pose = self.estimate_pose(img)
            if pose is None:
                invalid_count += 1
                continue
            pitch, yaw, _ = pose
            # if abs(pitch) > pitch_thresh or abs(yaw) > yaw_thresh:
            if abs(pitch) > pitch_thresh or abs(yaw) > yaw_thresh:
                invalid_count += 1
                # print(f"Discarding due to extreme pose for {os.path.basename(video_path)}: p: {pitch}, y: {yaw}")

        # invalid_ratio = invalid_count / total_frames
        invalid_ratio = invalid_count / (total_frames / 10)
        return invalid_ratio < max_invalid_ratio, invalid_ratio


    @torch.no_grad()
    def estimate_pose_from_frames(self, video_frames: np.ndarray, yaw_thresh: float = 45.0, pitch_thresh: float = 45.0, max_invalid_ratio: float = 0.5) -> bool:
        # video_frames = read_video(video_path, change_fps=False)
        video_frames = np.transpose(video_frames, (0, 3, 1, 2))  # (F, C, H, W)
        total_frames = len(video_frames)
        invalid_count = 0

        for frame in video_frames[::10]:
            img = np.transpose(frame, (1, 2, 0))  # CHW → HWC
            pose = self.estimate_pose(img)
            if pose is None:
                invalid_count += 1
                continue
            pitch, yaw, _ = pose
            # if abs(pitch) > pitch_thresh or abs(yaw) > yaw_thresh:
            if abs(pitch) > pitch_thresh or abs(yaw) > yaw_thresh:
                invalid_count += 1
                # print(f"Discarding due to extreme pose for {os.path.basename(video_path)}: p: {pitch}, y: {yaw}")

        # invalid_ratio = invalid_count / total_frames
        invalid_ratio = invalid_count / (total_frames / 10)
        return invalid_ratio < max_invalid_ratio, invalid_ratio


    @torch.no_grad()
    def estimate_landmark_motion_from_frames(
        self, 
        video_frames: np.ndarray, 
        static_thresh: float = 0.02, 
        max_static_ratio: float = 0.8, 
        cache_landmarks: bool = False):
        """
        Returns (is_dynamic, static_ratio) or (is_dynamic, static_ratio, cached_landmarks) if cache_landmarks=True
        """
        video_frames = np.transpose(video_frames, (0, 3, 1, 2))  # (F, C, H, W)
        motions = []
        prev_lmk = None
        static_count = 0
        frame_count = 0
        cached_landmarks = [] if cache_landmarks else None

        for frame_idx, frame in enumerate(video_frames[::5]):
            img = np.transpose(frame, (1, 2, 0))
            landmarks = self.fa.get_landmarks(img)
            if landmarks is None or len(landmarks) == 0:
                if cache_landmarks:
                    cached_landmarks.append({
                        "frame_idx": frame_idx * 5,
                        "landmarks": None,
                        "timestamp": frame_idx * 5 / 25.0  # Assuming 25 FPS
                    })
                continue

            current_lmk = landmarks[0]
            
            if cache_landmarks:
                cached_landmarks.append({
                    "frame_idx": frame_idx * 5,
                    "landmarks": current_lmk.tolist(),  # Convert numpy array to list for JSON serialization
                    "timestamp": frame_idx * 5 / 25.0  # Assuming 25 FPS
                })
            
            if prev_lmk is not None:
                _, _, disparity = procrustes(prev_lmk, current_lmk)
                motion_metric = np.sqrt(disparity)
                motions.append(motion_metric)
                if motion_metric < static_thresh:  # e.g., 0.01–0.02
                    static_count += 1
                    # print(f"## Static frame detected at {video_input} index {frame_count} with motion {motion_metric:.4f}")
            prev_lmk = current_lmk
            frame_count += 1

        if not motions or frame_count == 0:
            if cache_landmarks:
                return False, 1.0, cached_landmarks  # Consider it static by default
            return False, 1.0  # Consider it static by default

        static_ratio = static_count / len(motions)
        is_dynamic = static_ratio < max_static_ratio
        
        if cache_landmarks:
            return is_dynamic, static_ratio, cached_landmarks
        return is_dynamic, static_ratio

# if __name__ == "__main__":
#     video_processor = VideoProcessor(256, "cuda")
#     video_frames = video_processor.affine_transform_video("assets/demo2_video.mp4")
#     write_video("output.mp4", video_frames, fps=25)
