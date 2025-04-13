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
import sys
from typing import Any, Generator
import pytz
from datetime import datetime

def get_est_timestamp():
    """Get a timezone-aware timestamp in EST"""
    est = pytz.timezone('America/New_York')
    dt = datetime.now(est)
    return dt.strftime('%Y%m%d%H%M%S%f')


# Configure logging to output to stdout with appropriate level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

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
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "object_only_videos")
            write_object_only_video(out_dir, inference_api, session_id, all_frames_with_masks)
        except Exception as e:
            logger.exception(f"Error generating object-only video: {str(e)}")

# --------------- HELPER FUNCTIONS ---------------
def write_object_only_video(out_dir, inference_api, session_id, all_frames_with_masks):
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
    
    # Get the original video dimensions
    video_height = inference_state.get("video_height")
    video_width = inference_state.get("video_width")
    if not video_height or not video_width:
        logger.warning("Original video dimensions not found, using default")
        video_height, video_width = 1080, 1920
        
    # Get the FPS from the inference state or default to 30
    fps = inference_state.get("fps", 30)
    
    # Create a directory to store the debug videos
    os.makedirs(out_dir, exist_ok=True)
    
    # Create session-specific directory using timestamp
    timestamp_dir = get_est_timestamp()
    debug_dir = os.path.join(out_dir, timestamp_dir)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create temporary directories for frames and output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Sort frames by index to ensure correct order
        all_frames_with_masks.sort(key=lambda x: x['frame_index'])
        
        # Extract video information using ffmpeg
        try:
            logger.info(f"Processing original video: {video_path}")
            
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
        
        # Extract all frames directly for processing
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
            
            # List all extracted frame files
            frame_files = sorted(glob.glob(os.path.join(all_frames_dir, "frame_*.png")))
            logger.info(f"Extracted {len(frame_files)} frames from video")
        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            raise ValueError(f"Failed to extract frames: {e}")
        
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
            
            # Verify frame dimensions are reasonable
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                logger.warning(f"Frame {frame_idx} has unusual dimensions: {width}x{height}")
                # Simple dimension fix: use the dimensions we already know are correct
                try:
                    if width < 10 or height < 10:
                        # Use the dimensions we got from ffprobe earlier
                        logger.info(f"Correcting frame dimensions to match video properties")
                        # Create a new frame with the correct dimensions by resizing to video dimensions
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    logger.error(f"Error fixing frame dimensions: {e}")
            
            # Create empty mask with the verified dimensions
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Combine all object masks
            for mask_data in frame_data['results']:
                try:
                    import pycocotools.mask as mask_util
                    # Process only RLE format masks
                    obj_mask = None
                    
                    # Only handle RLE format masks
                    if 'mask' in mask_data and isinstance(mask_data['mask'], dict) and 'size' in mask_data['mask'] and 'counts' in mask_data['mask']:
                        logger.info(f"Processing RLE mask for frame {frame_idx}")
                        rle = mask_data['mask']
                        
                        # Debug: Print RLE mask details
                        logger.info(f"RLE mask size: {rle['size']}, counts length: {len(rle['counts'])}")
                        
                        # CRITICAL FIX: Check if the frame dimensions match the RLE mask dimensions
                        # The model may be using square frames internally but returning masks for the original video dimensions
                        rle_height, rle_width = rle['size']
                        
                        # Decode RLE to binary mask
                        obj_mask = mask_util.decode(rle)
                        
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
                    else:
                        logger.warning(f"Couldn't extract mask from format: {mask_data.keys()}")
                except Exception as e:
                    logger.error(f"Error decoding mask: {e}", exc_info=True)
            
            frame_count += 1
            
            # Apply the object-only effect
            try:
                # Convert mask to binary
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                
                # Create a normalized mask (0 or 1) for RGB channels
                normalized_mask = binary_mask / 255.0
                rgb_mask = np.stack([normalized_mask] * 3, axis=-1)
                
                # Create a 4-channel image (RGBA)
                rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                
                # Apply the mask to the RGB channels
                normalized_mask = binary_mask / 255.0
                rgb_mask = np.stack([normalized_mask] * 3, axis=-1)
                
                # Apply RGB mask
                rgba[:, :, 0:3] = rgba[:, :, 0:3] * rgb_mask
                
                # Set alpha channel (255 where object, 0 elsewhere)
                rgba[:, :, 3] = binary_mask
                
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
                
                # Save the frame directly to the timestamp directory with the requested naming pattern
                output_frame_path = os.path.join(debug_dir, f"frame_{frame_idx}.png")
                cv2.imwrite(output_frame_path, rgba)
                
                # Save mask frame with the requested naming pattern
                mask_frame = np.zeros((rgba.shape[0], rgba.shape[1]), dtype=np.uint8)
                mask_frame[binary_mask > 0] = 255
                
                # Save mask to the timestamp directory with the requested naming pattern
                mask_output_path = os.path.join(debug_dir, f"frame_{frame_idx}.png.mask.png")
                cv2.imwrite(mask_output_path, mask_frame)
                
                if frame_idx % 10 == 0:  # Log every 10th frame
                    logger.info(f"Processed frame {frame_idx}")
            except Exception as e:
                logger.exception(f"Error processing frame {frame_idx}: {str(e)}")
        
        logger.info(f"Processed {frame_count} frames successfully")
        
        if frame_count == 0:
            logger.error("No frames were processed")


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
