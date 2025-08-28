#!/usr/bin/env python3
"""
Debug visualization functions
Used to display numpy image data, helping debug face cropping ratio issues
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import decord
from typing import List, Tuple, Optional, Union
from pyffprobe import FFProbe
from decimal import Decimal, getcontext

getcontext().prec = 25

def show_image(image: np.ndarray, 
               title: str = "Image", 
               figsize: Tuple[int, int] = (10, 8),
               cmap: Optional[str] = None,
               show_bbox: Optional[Tuple[int, int, int, int]] = None,
               show_points: Optional[np.ndarray] = None,
               save_path: Optional[str] = None) -> None:
    """
    Display a single numpy image
    
    Args:
        image: numpy array image (H, W, C) or (H, W)
        title: image title
        figsize: image display size (width, height)
        cmap: color map (for grayscale images)
        show_bbox: display bounding box (x1, y1, x2, y2)
        show_points: display keypoints (N, 2) or (N, 3)
        save_path: save path (optional)
    """
    plt.figure(figsize=figsize)
    
    # Determine color map
    if len(image.shape) == 2:
        # Grayscale image
        plt.imshow(image, cmap=cmap or 'gray')
    else:
        # Color image
        if image.dtype == np.uint8:
            # If uint8, convert to 0-1 range
            image_display = image.astype(np.float32) / 255.0
        else:
            image_display = image
        plt.imshow(image_display)
    
    # Display bounding box
    if show_bbox is not None:
        x1, y1, x2, y2 = show_bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x1, y1-5, f'({x1},{y1})', color='red', fontsize=10)
        plt.text(x2, y2+15, f'({x2},{y2})', color='red', fontsize=10)
    
    # Display keypoints
    if show_points is not None:
        if show_points.shape[1] >= 2:
            x_coords = show_points[:, 0]
            y_coords = show_points[:, 1]
            plt.scatter(x_coords, y_coords, c='blue', s=20, alpha=0.7)
            # Label first few point indices
            for i in range(min(5, len(show_points))):
                plt.annotate(f'{i}', (x_coords[i], y_coords[i]), 
                           color='blue', fontsize=8)
    
    plt.title(title)
    plt.axis('on')  # Show coordinate axes
    plt.grid(True, alpha=0.3)
    
    # Add size information
    h, w = image.shape[:2]
    plt.xlabel(f'Width: {w}px')
    plt.ylabel(f'Height: {h}px')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    plt.show()

def show_image_comparison(images: List[np.ndarray], 
                         titles: List[str],
                         figsize: Tuple[int, int] = (15, 8),
                         show_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
                         show_points_list: Optional[List[np.ndarray]] = None) -> None:
    """
    Display multiple images side by side for comparison
    
    Args:
        images: list of images
        titles: list of titles
        figsize: image display size
        show_bboxes: list of bounding boxes
        show_points_list: list of keypoints
    """
    n_images = len(images)
    if n_images == 0:
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = axes[i]
        
        # Display image
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        else:
            if image.dtype == np.uint8:
                image_display = image.astype(np.float32) / 255.0
            else:
                image_display = image
            ax.imshow(image_display)
        
        # Display bounding box
        if show_bboxes and i < len(show_bboxes):
            bbox = show_bboxes[i]
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
        
        # Display keypoints
        if show_points_list and i < len(show_points_list):
            points = show_points_list[i]
            if points is not None and len(points) > 0:
                if points.shape[1] >= 2:
                    x_coords = points[:, 0]
                    y_coords = points[:, 1]
                    ax.scatter(x_coords, y_coords, c='blue', s=15, alpha=0.7)
        
        ax.set_title(f"{title}\n({image.shape[1]}x{image.shape[0]})")
        ax.axis('on')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def show_face_crop_debug(original_frame: np.ndarray,
                         detected_bbox: Tuple[int, int, int, int],
                         final_crop_coords: Tuple[int, int, int, int],
                         cropped_frames: np.ndarray,
                         padded_frames: np.ndarray,
                         resized_frames: np.ndarray) -> None:
    """
    Function specifically for debugging face cropping process
    
    Args:
        original_frame: original frame
        detected_bbox: detected face bounding box (x1, y1, x2, y2)
        final_crop_coords: final crop coordinates (x1, y1, x2, y2)
        cropped_frames: cropped frames
        padded_frames: padded frames
        resized_frames: resized frames
    """
    # Calculate size information
    orig_h, orig_w = original_frame.shape[:2]
    bbox_w = detected_bbox[2] - detected_bbox[0]
    bbox_h = detected_bbox[3] - detected_bbox[1]
    crop_w = final_crop_coords[2] - final_crop_coords[0]
    crop_h = final_crop_coords[3] - final_crop_coords[1]
    
    # Create debug information
    debug_info = f"""
    Original frame size: {orig_w} x {orig_h}
    Detected face: {bbox_w} x {bbox_h} (ratio: {bbox_w/bbox_h:.2f})
    Crop region: {crop_w} x {crop_h} (ratio: {crop_w/crop_h:.2f})
    Cropped frame count: {len(cropped_frames)}
    Padded size: {padded_frames.shape[1]} x {padded_frames.shape[2]}
    Final size: {resized_frames.shape[1]} x {resized_frames.shape[2]}
    """
    
    print("🔍 Face cropping debug information:")
    print(debug_info)
    
    # Display comparison images
    images = [
        original_frame,
        cropped_frames[0] if len(cropped_frames) > 0 else np.zeros((100, 100, 3)),
        padded_frames[0] if len(padded_frames) > 0 else np.zeros((100, 100, 3)),
        resized_frames[0] if len(resized_frames) > 0 else np.zeros((100, 100, 3))
    ]
    
    titles = [
        f"Original frame\n{orig_w}x{orig_h}",
        f"Cropped\n{crop_w}x{crop_h}",
        f"Padded\n{padded_frames.shape[1]}x{padded_frames.shape[2]}",
        f"Final output\n{resized_frames.shape[1]}x{resized_frames.shape[2]}"
    ]
    
    bboxes = [
        detected_bbox,
        None,
        None,
        None
    ]
    
    show_image_comparison(images, titles, figsize=(20, 5), show_bboxes=bboxes)

def show_landmarks_debug(frame: np.ndarray, 
                        landmarks: np.ndarray,
                        title: str = "Landmarks Debug") -> None:
    """
    Display keypoint debug information
    
    Args:
        frame: input frame
        landmarks: keypoint array (N, 2) or (N, 3)
        title: title
    """
    if landmarks is None or len(landmarks) == 0:
        print("⚠️ No keypoints detected")
        return
    
    print(f"🔍 Keypoint debug information:")
    print(f"  Detected {len(landmarks)} keypoints")
    print(f"  Keypoint shape: {landmarks.shape}")
    
    # Display coordinates of first few keypoints
    for i in range(min(5, len(landmarks))):
        point = landmarks[i]
        print(f"  Point {i}: ({point[0]:.1f}, {point[1]:.1f})")
    
    # Display image and keypoints
    show_image(frame, title, show_points=landmarks)

def show_batch_debug(frames: np.ndarray, 
                    title: str = "Batch Debug",
                    max_display: int = 4) -> None:
    """
    Display batch frame debug information
    
    Args:
        frames: batch frames (B, H, W, C)
        title: title
        max_display: maximum display frame count
    """
    if len(frames) == 0:
        print("⚠️ No frame data")
        return
    
    print(f"🔍 Batch debug information:")
    print(f"  Batch size: {len(frames)}")
    print(f"  Frame size: {frames.shape[1]} x {frames.shape[2]}")
    print(f"  Data type: {frames.dtype}")
    print(f"  Value range: [{frames.min():.2f}, {frames.max():.2f}]")
    
    # Select frames to display
    n_frames = min(max_display, len(frames))
    selected_frames = frames[:n_frames]
    
    # Create titles
    titles = [f"Frame {i}\n{frames.shape[1]}x{frames.shape[2]}" 
              for i in range(n_frames)]
    
    show_image_comparison(selected_frames, titles, figsize=(15, 4))

def fix_video_aspect_ratio(
    video_path,
    show_image=False,
    # to_sar=True,
):
    """
    Check and fix video frame display ratio.
    If the video's pixel aspect ratio (SAR) is not equal to 1:1, perform stretching.

    Args:
        video_path (str): Path to video file.
        show_image (bool): Whether to display image.
    """
    try:
        # Use pyffprobe to get video stream information
        ffprobe = FFProbe(video_path)
        video_stream = ffprobe.video[0] # Get first video stream
        
        # Get video pixel aspect ratio (SAR)
        sar_str = video_stream.sample_aspect_ratio
        if not sar_str:
            # print("No SAR information found")
            raise ValueError("No SAR information found")
        
        # Parse SAR string, e.g., '9:16'
        sar_parts = sar_str.split(':')
        sar_w_ratio = int(sar_parts[0])
        sar_h_ratio = int(sar_parts[1])
        video_sar = float(sar_w_ratio / sar_h_ratio)
        
        # Get original video width and height
        original_width = int(video_stream.width)
        original_height = int(video_stream.height)
        
        # If video width and height are different, but SAR ratio is 1:1, then video ratio is normal, no need to fix
        # But if video width and height are the same, but SAR ratio is not 1:1, then video encoding stretched pixels, video data itself is abnormal, needs fixing
        if original_width != original_height and \
            Decimal(str(video_sar)) == Decimal(str(1)):
            # print("Video ratio is normal, no need to fix.")
            return None, None

        # Calculate target stretch size based on SAR
        # Target width = original height * SAR
        target_width = int(original_height * (sar_w_ratio / sar_h_ratio))
        # Target height = original width / SAR
        target_height = int(original_width / (sar_w_ratio / sar_h_ratio))

        # Intelligently select stretch dimensions to ensure stretched image won't be compressed
        if target_width >= original_width:
            new_width = target_width
            new_height = original_height
        else:
            new_width = original_width
            new_height = target_height
        
        return new_width, new_height
        
        # Determine interpolation method
        # As long as new size has one dimension larger than original dimension, consider it as enlargement
        if new_width > original_width or new_height > original_height:
            interpolation_method = cv2.INTER_CUBIC
            print("Using INTER_CUBIC method for enlargement interpolation.")
        else:
            interpolation_method = cv2.INTER_AREA
            print("Using INTER_AREA method for reduction interpolation.")

        # Use decord to read video
        vr = decord.VideoReader(video_path)

        for i in range(len(vr)):
            # Read current frame and convert to NumPy array
            frame = vr[i].asnumpy()

            # Use OpenCV resize function to stretch frame
            # INTER_CUBIC is a high-quality interpolation method
            resized_frame = cv2.resize(
                frame,
                (new_width, new_height),
                interpolation=interpolation_method
            )

            # Here you can perform subsequent processing on resized_frame, such as display or save
            if show_image:
                cv2.imshow('Fixed Video Frame', resized_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Error occurred while processing video: {e}")

    finally:
        # Ensure window is properly destroyed
        if 'vr' in locals():
            del vr
        if show_image:
            cv2.destroyAllWindows()

# Example usage function
def example_usage():
    """Usage examples"""
    print("📖 Debug visualization function usage examples:")
    print()
    
    print("1. Display single image:")
    print("   show_image(image, 'Face detection result')")
    print()
    
    print("2. Display image with bounding box:")
    print("   show_image(image, 'Crop region', show_bbox=(100, 100, 300, 400))")
    print()
    
    print("3. Display image with keypoints:")
    print("   show_image(image, 'Keypoints', show_points=landmarks)")
    print()
    
    print("4. Compare multiple images:")
    print("   show_image_comparison([img1, img2], ['Original', 'Processed'])")
    print()
    
    print("5. Face cropping debug:")
    print("   show_face_crop_debug(original, bbox, crop_coords, cropped, padded, resized)")
    print()
    
    print("6. Keypoint debug:")
    print("   show_landmarks_debug(frame, landmarks)")
    print()
    
    print("7. Batch debug:")
    print("   show_batch_debug(frames, 'Process batch')")
    print()

if __name__ == "__main__":
    # example_usage()
    fix_video_aspect_ratio(
        video_path="/mnt/425b37ab-69e4-4f4d-a719-d3f3d3a7c387/behavior_s2s_data/seamless_interaction_yllee/improvised/train/0000/0038/V00_S0039_I00000581_P0040A.mp4"
    )
