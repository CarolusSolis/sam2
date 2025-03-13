# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import shutil
import json
import glob
import os
import re
import subprocess
import tempfile
import numpy as np
import cv2
from typing import Any, Generator

from app_conf import (
    GALLERY_PATH,
    GALLERY_PREFIX,
    POSTERS_PATH,
    POSTERS_PREFIX,
    UPLOADS_PATH,
    UPLOADS_PREFIX,
)
from data.loader import preload_data
from data.schema import schema
from data.store import set_videos
from flask import Flask, make_response, Request, request, Response, send_from_directory
from flask_cors import CORS
from inference.data_types import PropagateDataResponse, PropagateInVideoRequest
from inference.multipart import MultipartResponseBuilder
from inference.predictor import InferenceAPI
from strawberry.flask.views import GraphQLView

logger = logging.getLogger(__name__)

app = Flask(__name__)
cors = CORS(app, supports_credentials=True)

videos = preload_data()
set_videos(videos)

inference_api = InferenceAPI()


@app.route("/healthy")
def healthy() -> Response:
    return make_response("OK", 200)


@app.route(f"/{GALLERY_PREFIX}/<path:path>", methods=["GET"])
def send_gallery_video(path: str) -> Response:
    try:
        return send_from_directory(
            GALLERY_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{POSTERS_PREFIX}/<path:path>", methods=["GET"])
def send_poster_image(path: str) -> Response:
    try:
        return send_from_directory(
            POSTERS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{UPLOADS_PREFIX}/<path:path>", methods=["GET"])
def send_uploaded_video(path: str):
    try:
        return send_from_directory(
            UPLOADS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


# TOOD: Protect route with ToS permission check
@app.route("/propagate_in_video", methods=["POST"])
def propagate_in_video() -> Response:
    data = request.json
    args = {
        "session_id": data["session_id"],
        "start_frame_index": data.get("start_frame_index", 0),
    }

    boundary = "frame"
    frame = gen_track_with_mask_stream(boundary, **args)
    return Response(frame, mimetype="multipart/x-savi-stream; boundary=" + boundary)


def gen_track_with_mask_stream(
    boundary: str,
    session_id: str,
    start_frame_index: int,
) -> Generator[bytes, None, None]:
    import os
    import cv2
    import numpy as np
    import tempfile
    import shutil
    import subprocess
    import time
    import torch
    import re
    import decord
    from decord import VideoReader
    
    # Store all masks and frames for later processing
    all_frames_with_masks = []
    
    with inference_api.autocast_context():
        request = PropagateInVideoRequest(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=start_frame_index,
        )

        # Collect all masks during propagation
        logger.info(f"Starting propagation for session {session_id}")
        for chunk in inference_api.propagate_in_video(request=request):
            # Store the frame index and mask data for later processing
            frame_data = chunk.to_dict()
            all_frames_with_masks.append(frame_data)
            
            # Yield the chunk to the client as normal
            yield MultipartResponseBuilder.build(
                boundary=boundary,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Frame-Current": "-1",
                    # Total frames minus the reference frame
                    "Frame-Total": "-1",
                    "Mask-Type": "RLE[]",
                },
                body=chunk.to_json().encode("UTF-8"),
            ).get_message()
        
        # After propagation is complete, generate the object-only video
        logger.info(f"Propagation complete for session {session_id}, generating object-only video")
        
        try:
            # Get the session state directly from the inference API's session_states dictionary
            session_data = inference_api.session_states.get(session_id)
            if not session_data:
                logger.error(f"No session found for session ID: {session_id}")
                return
                
            # Access the state from the session data
            inference_state = session_data.get("state")
            if not inference_state:
                logger.error(f"No inference state found in session: {session_id}")
                return
            
            # Get the original video path from the inference state
            video_path = inference_state.get("video_path")
            if not video_path:
                logger.error("Original video path not found in inference state")
                raise ValueError("Cannot access original video path - required for object-only video generation")
                
            # We have the original video path, set up to read frames directly from it
            logger.info(f"Using original video for masked object output: {video_path}")
            # Get the image size used in the model
            image_size = inference_state.get("image_size", 1024)
            
            # Get the original video dimensions
            video_height = inference_state.get("video_height")
            video_width = inference_state.get("video_width")
            if not video_height or not video_width:
                logger.warning("Original video dimensions not found, using default")
                video_height, video_width = 1080, 1920
                
            # Get the FPS from the inference state or default to 30
            fps = inference_state.get("fps", 30)
            
            # Create a directory to store the debug videos
            debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_videos")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create temporary directories for frames and output
            with tempfile.TemporaryDirectory() as temp_dir:
                frames_dir = os.path.join(temp_dir, "frames")
                os.makedirs(frames_dir, exist_ok=True)
                
                # Sort frames by index to ensure correct order
                all_frames_with_masks.sort(key=lambda x: x['frame_index'])
                
                # Extract original frames using ffmpeg
                try:
                    # Extract 5 frames for initial testing/debugging
                    logger.info(f"Extracting frames from original video: {video_path}")
                    extract_cmd = [
                        "ffmpeg",
                        "-i", video_path,  # Input video
                        "-vf", "select='eq(n,0)+eq(n,1)+eq(n,2)+eq(n,3)+eq(n,4)'",  # Just get first 5 frames for testing
                        "-vsync", "0",     # Prevent frame dropping
                        "-q:v", "1",       # High quality
                        os.path.join(debug_dir, "original_frame_%03d.png")  # Output pattern
                    ]
                    subprocess.run(extract_cmd, check=True)
                    logger.info("Successfully extracted test frames")
                    
                    # Get video properties using ffprobe
                    probe_cmd = [
                        "ffprobe", 
                        "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "json",
                        video_path
                    ]
                    probe_output = json.loads(subprocess.check_output(probe_cmd, text=True))
                    video_info = probe_output["streams"][0]
                    
                    # Get dimensions and FPS
                    width = int(video_info["width"])
                    height = int(video_info["height"])
                    fps_str = video_info["r_frame_rate"]
                    if '/' in fps_str:
                        num, den = map(int, fps_str.split('/'))
                        fps = num / den
                    else:
                        fps = float(fps_str)
                        
                    logger.info(f"Original video dimensions: {width}x{height}, FPS: {fps}")
                except Exception as e:
                    logger.error(f"Error extracting frames from original video: {e}")
                    raise ValueError(f"Failed to extract frames: {e}")
                
                # Now extract all frames for processing
                logger.info("Extracting all frames from video for mask application")
                all_frames_dir = os.path.join(temp_dir, "all_frames")
                os.makedirs(all_frames_dir, exist_ok=True)
                
                # Extract all frames with consistent naming pattern
                extract_all_cmd = [
                    "ffmpeg",
                    "-i", video_path,  # Input video
                    "-vsync", "0",     # Prevent frame dropping
                    "-q:v", "1",       # High quality
                    os.path.join(all_frames_dir, "frame_%04d.png")  # Output pattern
                ]
                try:
                    subprocess.run(extract_all_cmd, check=True)
                    logger.info("Successfully extracted all frames for processing")
                except Exception as e:
                    logger.error(f"Error extracting all frames: {e}")
                    raise ValueError(f"Failed to extract all frames: {e}")
                
                # List all extracted frame files
                frame_files = sorted(glob.glob(os.path.join(all_frames_dir, "frame_*.png")))
                logger.info(f"Extracted {len(frame_files)} frames from video")
                
                # Generate frames with object-only effect
                frame_count = 0
                
                for frame_data in all_frames_with_masks:
                    frame_idx = frame_data['frame_index']
                    
                    # Ensure frame index is within bounds
                    if frame_idx >= len(frame_files):
                        logger.warning(f"Frame index {frame_idx} out of bounds (max: {len(frame_files)-1})")
                        continue
                    
                    # Get the matching frame file
                    frame_file = frame_files[frame_idx]
                    
                    try:
                        # Read the extracted frame directly
                        frame = cv2.imread(frame_file)
                        if frame is None:
                            logger.error(f"Failed to read frame from {frame_file}")
                            continue
                            
                        # Debug first few frames
                        if frame_idx < 3:
                            logger.info(f"Processing frame {frame_idx} from {frame_file}")
                            # Save a copy of the original frame
                            cv2.imwrite(os.path.join(debug_dir, f"direct_frame_{frame_idx}.png"), frame)
                    except Exception as e:
                        logger.error(f"Error reading frame {frame_idx} from extracted files: {e}")
                        continue
                    
                    # Ensure we have 3 channels for OpenCV processing
                    if len(frame.shape) == 2:  # Grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:  # RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[2] != 3:
                        logger.warning(f"Unexpected number of channels: {frame.shape}")
                        continue
                    
                    # Create a mask from the results
                    # Get frame dimensions from the frame
                    height, width = frame.shape[:2]
                    
                    # Debug frame dimensions for the first few frames
                    if frame_idx < 3:
                        print(f"Frame {frame_idx} - Original frame dimensions: {width}x{height}")
                        # Save frame shape to debug file
                        with open(os.path.join(debug_dir, f"frame_{frame_idx}_info.txt"), 'w') as f:
                            f.write(f"Frame dimensions: {width}x{height}\n")
                            f.write(f"Frame shape: {frame.shape}\n")
                            f.write(f"Frame dtype: {frame.dtype}\n")
                    
                    # CRITICAL: Verify frame dimensions are reasonable
                    # Check if frame dimensions are extremely unbalanced (like 4x1280)
                    aspect_ratio = width / height if height > 0 else 0
                    if aspect_ratio < 0.1 or aspect_ratio > 10:
                        logger.warning(f"Frame {frame_idx} has unusual dimensions: {width}x{height}, aspect ratio: {aspect_ratio:.2f}")
                        # If frame is extremely narrow, try to fix it by loading the original video frame
                        try:
                            # This is a fallback to ensure we have reasonable frame dimensions
                            if width < 10 or height < 10:
                                logger.warning(f"Frame {frame_idx} dimensions too small, attempting to fix")
                                # For debugging, save the problematic frame
                                cv2.imwrite(os.path.join(debug_dir, f"problematic_frame_{frame_idx}.png"), frame)
                                
                                # CRITICAL FIX: Instead of assuming 16:9, get the original video dimensions
                                # This ensures we maintain the same proportions as the input video
                                try:
                                    # Get original video dimensions using OpenCV
                                    cap = cv2.VideoCapture(video_path)
                                    if cap.isOpened():
                                        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                        orig_aspect = orig_width / orig_height if orig_height > 0 else 16/9
                                        cap.release()
                                        logger.info(f"Original video dimensions: {orig_width}x{orig_height}, aspect ratio: {orig_aspect:.2f}")
                                    else:
                                        # Fallback to 16:9 if can't open video
                                        orig_aspect = 16/9
                                        logger.warning(f"Could not open video to get dimensions, using default aspect ratio: {orig_aspect}")
                                except Exception as e:
                                    # Fallback to 16:9 if error
                                    orig_aspect = 16/9
                                    logger.warning(f"Error getting video dimensions: {e}, using default aspect ratio: {orig_aspect}")
                                
                                # Fix dimensions based on original aspect ratio
                                if width < 10 and height > 100:
                                    # Likely a tall, narrow frame that should be wider
                                    new_width = int(height * orig_aspect)  # Use original aspect ratio
                                    logger.info(f"Adjusting frame width from {width} to {new_width} using original aspect ratio {orig_aspect:.2f}")
                                    width = new_width
                                    # Create a new frame with the correct dimensions by resizing
                                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                                    # Save the resized frame for debugging
                                    cv2.imwrite(os.path.join(debug_dir, f"resized_frame_{frame_idx}.png"), frame)
                                elif height < 10 and width > 100:
                                    # Likely a wide, short frame that should be taller
                                    new_height = int(width / orig_aspect)  # Use original aspect ratio
                                    logger.info(f"Adjusting frame height from {height} to {new_height} using original aspect ratio {orig_aspect:.2f}")
                                    height = new_height
                                    # Create a new frame with the correct dimensions by resizing
                                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                                    # Save the resized frame for debugging
                                    cv2.imwrite(os.path.join(debug_dir, f"resized_frame_{frame_idx}.png"), frame)
                        except Exception as e:
                            logger.error(f"Error fixing frame dimensions: {e}")
                    
                    # Create empty mask with the verified dimensions
                    mask = np.zeros((height, width), dtype=np.uint8)
                    
                    # Combine all object masks
                    for mask_data in frame_data['results']:
                        try:
                            import pycocotools.mask as mask_util
                            # Process only RLE format masks - the simple, elegant solution
                            obj_mask = None
                            
                            # Only handle RLE format masks
                            if 'mask' in mask_data and isinstance(mask_data['mask'], dict) and 'size' in mask_data['mask'] and 'counts' in mask_data['mask']:
                                logger.info(f"Processing RLE mask for frame {frame_idx}")
                                rle = mask_data['mask']
                                
                                # Debug: Print RLE mask details
                                logger.info(f"RLE mask size: {rle['size']}, counts length: {len(rle['counts'])}")
                                
                                # Save RLE details for first few frames
                                if frame_idx < 3:
                                    print(f"Frame {frame_idx} - RLE: size={rle['size']}, counts (first 50 chars): {str(rle['counts'])[:50]}")
                                    
                                # CRITICAL FIX: Check if the frame dimensions match the RLE mask dimensions
                                # The model may be using square frames internally but returning masks for the original video dimensions
                                rle_height, rle_width = rle['size']
                                
                                # Debug the frame dimensions vs RLE dimensions
                                if frame_idx < 3:
                                    print(f"Frame {frame_idx} - Frame dimensions: {width}x{height}, RLE mask dimensions: {rle_width}x{rle_height}")
                                
                                # Decode RLE to binary mask
                                obj_mask = mask_util.decode(rle)
                                
                                # Debug: Save raw decoded mask for first few frames
                                if frame_idx < 3:
                                    debug_mask_path = os.path.join(debug_dir, f"raw_mask_{frame_idx}.png")
                                    cv2.imwrite(debug_mask_path, obj_mask * 255)
                                    print(f"Saved raw mask for frame {frame_idx} to {debug_mask_path}")
                                    print(f"Raw mask stats - min: {obj_mask.min()}, max: {obj_mask.max()}, mean: {obj_mask.mean():.4f}")
                                
                                obj_mask = obj_mask * 255
                                
                                # Get the mask dimensions
                                mask_height, mask_width = obj_mask.shape[:2]
                                
                                # CRITICAL FIX: Always ensure the mask matches the actual frame dimensions
                                # This is essential because the model may be using different dimensions internally
                                if mask_height != height or mask_width != width:
                                    logger.info(f"Resizing mask from {mask_width}x{mask_height} to {width}x{height}")
                                    obj_mask = cv2.resize(obj_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                                    
                                # Debug: Check if mask has any non-zero values after processing
                                if frame_idx < 3 or frame_idx % 50 == 0:
                                    non_zero = np.count_nonzero(obj_mask)
                                    logger.info(f"Frame {frame_idx} - Mask has {non_zero} non-zero pixels out of {obj_mask.size} ({non_zero/obj_mask.size*100:.2f}%)")
                            
                            # Update the combined mask if we found a valid object mask
                            if obj_mask is not None:
                                # Ensure mask is binary
                                if obj_mask.max() > 1:
                                    _, obj_mask = cv2.threshold(obj_mask, 127, 255, cv2.THRESH_BINARY)
                                
                                # Final validation: Ensure mask has right dimensions
                                # This should never happen at this point, but double-check as a precaution
                                if obj_mask.shape[:2] != (height, width):
                                    logger.warning(f"Final mask shape {obj_mask.shape} doesn't match frame shape {(height, width)}")
                                    # Resize mask if needed
                                    obj_mask = cv2.resize(obj_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                                
                                # Combine with existing mask
                                mask = np.maximum(mask, obj_mask)
                                
                                # Debug: After combining masks
                                if frame_idx < 3:
                                    non_zero_combined = np.count_nonzero(mask)
                                    logger.info(f"Frame {frame_idx} - After combining: mask has {non_zero_combined} non-zero pixels ({non_zero_combined/mask.size*100:.2f}%)")
                                    # Save the combined mask for debugging
                                    combined_mask_path = os.path.join(debug_dir, f"combined_mask_{frame_idx}.png")
                                    cv2.imwrite(combined_mask_path, mask)
                                    print(f"Saved combined mask for frame {frame_idx} to {combined_mask_path}")
                            else:
                                logger.warning(f"Couldn't extract mask from format: {mask_data.keys()}")
                        except Exception as e:
                            logger.error(f"Error decoding mask: {e}", exc_info=True)
                    
                    frame_count += 1
                    
                    # Apply the object-only effect
                    try:
                        # Debug: Before converting to binary
                        if frame_idx < 3:
                            print(f"Frame {frame_idx} - Before binary threshold: min={mask.min()}, max={mask.max()}, mean={mask.mean():.4f}")
                        
                        # Convert mask to binary
                        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                        
                        # Debug: Save binary mask for first few frames
                        if frame_idx < 3:
                            binary_mask_path = os.path.join(debug_dir, f"binary_mask_{frame_idx}.png")
                            cv2.imwrite(binary_mask_path, binary_mask)
                            print(f"Saved binary mask for frame {frame_idx} to {binary_mask_path}")
                            print(f"Binary mask stats - min: {binary_mask.min()}, max: {binary_mask.max()}, unique values: {np.unique(binary_mask)}")
                        
                        # Create a normalized mask (0 or 1) for RGB channels
                        normalized_mask = binary_mask / 255.0
                        rgb_mask = np.stack([normalized_mask] * 3, axis=-1)
                        
                        # Create a 4-channel image (RGBA)
                        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                        
                        # Save original frame for first few frames
                        if frame_idx < 3:
                            orig_frame_path = os.path.join(debug_dir, f"original_frame_{frame_idx}.png")
                            cv2.imwrite(orig_frame_path, frame)
                            print(f"Saved original frame {frame_idx} to {orig_frame_path}")
                        
                        # Apply the mask to the RGB channels
                        normalized_mask = binary_mask / 255.0
                        rgb_mask = np.stack([normalized_mask] * 3, axis=-1)
                        
                        # Save RGB mask for debugging
                        if frame_idx < 3:
                            # Convert rgb_mask to 8-bit format for saving
                            rgb_mask_vis = (rgb_mask * 255).astype(np.uint8)
                            rgb_mask_path = os.path.join(debug_dir, f"rgb_mask_{frame_idx}.png")
                            cv2.imwrite(rgb_mask_path, cv2.merge([rgb_mask_vis[:,:,2], rgb_mask_vis[:,:,1], rgb_mask_vis[:,:,0]]))
                            print(f"Saved RGB mask for frame {frame_idx} to {rgb_mask_path}")
                        
                        # Apply RGB mask
                        rgba[:, :, 0:3] = rgba[:, :, 0:3] * rgb_mask
                        
                        # Debug: Check RGB values after masking
                        if frame_idx < 3:
                            print(f"Frame {frame_idx} - After RGB masking: unique RGB channel values: {np.unique(rgba[:,:,0:3])}")
                        
                        # Set alpha channel (255 where object, 0 elsewhere)
                        rgba[:, :, 3] = binary_mask
                        
                        # Debug: Check alpha channel values
                        if frame_idx < 3:
                            print(f"Frame {frame_idx} - Alpha channel min: {rgba[:,:,3].min()}, max: {rgba[:,:,3].max()}, unique values: {np.unique(rgba[:,:,3])}")
                        
                        # Ensure dimensions are even for FFmpeg (yuv420p compatibility)
                        h, w = rgba.shape[:2]
                        if h % 2 != 0 or w % 2 != 0:
                            logger.info(f"Frame dimensions {w}x{h} not even, adjusting for FFmpeg compatibility")
                            # Create padded frame with even dimensions
                            new_h = h + (h % 2)  # Make height even by adding padding if needed
                            new_w = w + (w % 2)  # Make width even by adding padding if needed
                            padded_rgba = np.zeros((new_h, new_w, 4), dtype=rgba.dtype)
                            padded_rgba[:h, :w] = rgba
                            rgba = padded_rgba
                        
                        # Print debug info for the first few frames to check color values
                        if frame_idx < 3:
                            # Sample a point in the mask where the object is present (if possible)
                            y_coords, x_coords = np.where(binary_mask > 0)
                            if len(y_coords) > 0:
                                # Take the center point of the object
                                center_idx = len(y_coords) // 2
                                y, x = y_coords[center_idx], x_coords[center_idx]
                                
                                # Print original frame color at this point
                                print(f"Frame {frame_idx} - Original BGR color at object point ({x},{y}): {frame[y,x]}")
                                
                                # Print RGBA color after mask application
                                print(f"Frame {frame_idx} - RGBA color at same point after mask: {rgba[y,x]}")
                                
                                # Check a few more random points in the object
                                for i in range(min(3, len(y_coords))):
                                    sample_idx = (center_idx + (i+1)*len(y_coords)//4) % len(y_coords)
                                    sy, sx = y_coords[sample_idx], x_coords[sample_idx]
                                    print(f"Frame {frame_idx} - Sample {i+1}: Original BGR: {frame[sy,sx]}, RGBA: {rgba[sy,sx]}")
                        
                        # Save the frame
                        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
                        write_success = cv2.imwrite(frame_path, rgba)
                        
                        # Debug: Verify the frame was written successfully
                        if frame_idx < 3 or not write_success:
                            print(f"Frame {frame_idx} - cv2.imwrite success: {write_success}")
                            # Check if the file exists and has content
                            if os.path.exists(frame_path):
                                file_size = os.path.getsize(frame_path)
                                print(f"Frame {frame_idx} - Frame file exists, size: {file_size} bytes")
                                # For first few frames, save a copy to debug dir for easy inspection
                                if frame_idx < 3:
                                    debug_frame_path = os.path.join(debug_dir, f"final_frame_{frame_idx}.png")
                                    shutil.copy(frame_path, debug_frame_path)
                                    print(f"Copied frame {frame_idx} to {debug_frame_path}")
                            else:
                                print(f"Frame {frame_idx} - ERROR: Frame file does not exist!")
                        
                        if frame_idx % 10 == 0:  # Log every 10th frame
                            logger.info(f"Processed frame {frame_idx}")
                    except Exception as e:
                        logger.exception(f"Error processing frame {frame_idx}: {str(e)}")
                
                logger.info(f"Processed {frame_count} frames successfully for object-only video")
                
                if frame_count == 0:
                    logger.error("No frames were processed - cannot create object-only video")
                    return
                
                # List the frames to verify they exist
                frame_files = sorted(os.listdir(frames_dir))
                print(f"Found {len(frame_files)} frame files in temp directory")
                if not frame_files:
                    print("No frame files found in the temporary directory")
                    # Debug: Output the directory structure for troubleshooting
                    print(f"Directory structure for debugging:")
                    for root, dirs, files in os.walk(temp_dir):
                        logger.debug(f"Directory: {root}")
                        logger.debug(f"Files: {files}")
                    return
                    
                # Use a start index that actually exists
                first_frame = frame_files[0]
                try:
                    pattern_match = re.search(r'frame_(\d+)\.png', first_frame)
                    if pattern_match:
                        start_number = int(pattern_match.group(1))
                        logger.info(f"Detected start frame number: {start_number}")
                    else:
                        start_number = 0
                        logger.warning(f"Could not detect start number from {first_frame}, using 0")
                except Exception as e:
                    start_number = 0
                    logger.warning(f"Error parsing first frame name: {e}, using start_number=0")
                
                # Use FFmpeg to encode the frames to a high-quality video
                output_path = os.path.join(temp_dir, "output.mp4")
                debug_output_path = os.path.join(debug_dir, f"object_only_{session_id}.mp4")
                print(f"Encoding object-only video to: {debug_output_path}")
                
                # Get original video dimensions to ensure output matches input
                original_width = None
                original_height = None
                try:
                    # Get original video dimensions using OpenCV
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        # Ensure dimensions are even for yuv420p
                        original_width += original_width % 2
                        original_height += original_height % 2
                        cap.release()
                        logger.info(f"Original video dimensions: {original_width}x{original_height}")
                except Exception as e:
                    logger.warning(f"Error getting original video dimensions: {e}")
                
                # Use FFmpeg to create the final video with original dimensions and quality
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",                   # Overwrite output file if it exists
                    "-framerate", str(fps),  # Use the original video's framerate
                    "-i", os.path.join(frames_dir, "frame_%04d.png"),  # Input frames
                    "-c:v", "libx264",     # H.264 codec
                    "-crf", "17",          # High quality (0-51, lower is better)
                    "-preset", "medium",    # Better balance between speed and quality
                    "-pix_fmt", "yuv420p", # Standard pixel format for compatibility
                    output_path              # Output file
                ]
                
                logger.info(f"Creating final video with dimensions {width}x{height} at {fps} FPS")
                
                print(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
                try:
                    # Check input files before running FFmpeg
                    first_few_frames = sorted(os.listdir(frames_dir))[:5]
                    print(f"First few frames in directory: {first_few_frames}")
                    
                    # Check file sizes and dimensions of first few frames
                    for frame_file in first_few_frames:
                        frame_path = os.path.join(frames_dir, frame_file)
                        if os.path.exists(frame_path):
                            file_size = os.path.getsize(frame_path)
                            try:
                                img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                                if img is not None:
                                    print(f"Frame {frame_file}: size={file_size} bytes, dimensions={img.shape}, type={img.dtype}")
                                else:
                                    print(f"Frame {frame_file}: size={file_size} bytes, ERROR: cv2.imread returned None")
                            except Exception as e:
                                print(f"Error reading frame {frame_file}: {e}")
                    
                    # Run FFmpeg with more verbose output for debugging
                    print("Running FFmpeg command...")
                    result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                    print("FFmpeg encoding completed successfully")
                    print(f"FFmpeg stdout: {result.stdout}")
                    print(f"FFmpeg stderr: {result.stderr}")
                    
                    # Debug: Verify output file exists and has content
                    if os.path.exists(output_path):
                        output_size = os.path.getsize(output_path)
                        print(f"Output file exists, size: {output_size} bytes")
                        
                        # Try to extract a frame from the output to verify it's valid
                        try:
                            extract_cmd = [
                                "ffmpeg",
                                "-i", output_path,
                                "-vframes", "1",
                                "-f", "image2",
                                os.path.join(debug_dir, "output_frame.png")
                            ]
                            subprocess.run(extract_cmd, check=True, capture_output=True, text=True)
                            print("Successfully extracted a frame from the output video")
                        except Exception as e:
                            print(f"Error extracting frame from output video: {e}")
                    else:
                        print(f"ERROR: Output file {output_path} does not exist!")
                    
                    # Verify the output file exists and has size
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        # Save a copy to the debug directory
                        shutil.copy2(output_path, debug_output_path)
                        print(f"Saved object-only video to {debug_output_path}")
                    else:
                        print(f"Output file missing or empty: {output_path}")
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg error: {e.stderr}")
                    
        except Exception as e:
            logger.exception(f"Error generating object-only video: {str(e)}")


class MyGraphQLView(GraphQLView):
    def get_context(self, request: Request, response: Response) -> Any:
        return {"inference_api": inference_api}


# Add GraphQL route to Flask app.
app.add_url_rule(
    "/graphql",
    view_func=MyGraphQLView.as_view(
        "graphql_view",
        schema=schema,
        # Disable GET queries
        # https://strawberry.rocks/docs/operations/deployment
        # https://strawberry.rocks/docs/integrations/flask
        allow_queries_via_get=False,
        # Strawberry recently changed multipart request handling, which now
        # requires enabling support explicitly for views.
        # https://github.com/strawberry-graphql/strawberry/issues/3655
        multipart_uploads_enabled=True,
    ),
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
