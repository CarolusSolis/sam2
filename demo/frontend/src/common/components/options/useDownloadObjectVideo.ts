/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import {getFileName} from '@/common/components/options/ShareUtils';
import {VideoRef} from '@/common/components/video/Video';
import useVideo from '@/common/components/video/editor/useVideo';
import {useState} from 'react';

type DownloadingState = 'default' | 'started' | 'processing' | 'downloading' | 'completed' | 'error';

type State = {
  state: DownloadingState;
  progress: number;
  downloadObjectOnly: () => Promise<void>;
};

export default function useDownloadObjectVideo(): State {
  const [downloadingState, setDownloadingState] = useState<DownloadingState>('default');
  const [progress, setProgress] = useState<number>(0);
  const video = useVideo();

  async function downloadObjectOnly(): Promise<void> {
    if (downloadingState === 'started' || downloadingState === 'processing') {
      return; // Already in progress
    }

    try {
      setDownloadingState('started');
      setProgress(0);

      // Get the current session ID directly from the video component's worker bridge
      // The worker bridge stores the current session ID when startSession is called
      const videoObj = video;
      if (!videoObj) {
        throw new Error('Video component not available');
      }
      
      // We need to use a different approach to get the session ID
      // The video component has a startSession method that returns the session ID
      // We'll use a hack to get the current session ID by checking if there's an active session
      
      // First, let's try to get the current session from the video component directly
      // This approach doesn't rely on internal implementation details
      let sessionId: string | null = null;
      
      try {
        // Try a more direct approach to get the session ID
        const bridge: VideoRef = videoObj;
        
        // Try to access the worker directly to get the session ID
        // This is a bit of a hack, but it's more reliable than event listeners
        const anyBridge = bridge as any;
        if (anyBridge.worker && anyBridge.worker.sessionId) {
          sessionId = anyBridge.worker.sessionId;
          console.log('Found session ID directly from worker:', sessionId);
        } else {
          console.log('No session ID found directly on worker, trying alternative method');
          
          // Try to get the session ID by creating a tracklet
          const tracklet = await bridge.createTracklet();
          console.log('Created tracklet:', tracklet);
          
          // Check if the tracklet has a sessionId property
          if (tracklet && (tracklet as any).sessionId) {
            sessionId = (tracklet as any).sessionId;
            console.log('Got session ID from tracklet:', sessionId);
          } else {
            // As a last resort, try to use the logAnnotations method
            // This might trigger events that contain the session ID
            console.log('Trying to get session ID from annotations');
            const annotations = await bridge.logAnnotations();
            console.log('Annotations:', annotations);
            
            // Check if annotations contain session ID
            // TypeScript doesn't know the return type of logAnnotations, so we need to cast it
            const annotationsObj = annotations as any;
            if (annotationsObj && annotationsObj.sessionId) {
              sessionId = annotationsObj.sessionId;
              console.log('Got session ID from annotations:', sessionId);
            }
          }
        }
      } catch (e) {
        // If this fails, we don't have an active session
        console.error('Error getting session ID:', e);
      }
      
      if (!sessionId) {
        throw new Error('No active session found. Try tracking an object first.');  
      }

      // Pause the video during processing
      video?.pause();

      // Request the server to encode the object-only video
      setDownloadingState('processing');
      setProgress(25);
      
      console.log('Requesting object-only video with session ID:', sessionId);
      
      // Add a small delay to ensure any background processes have completed
      await new Promise(resolve => setTimeout(resolve, 500));
      
      try {
        console.log('Making request to /encode_video endpoint with session ID:', sessionId);
        const response = await fetch('/encode_video', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            session_id: sessionId,
            effect: 'object_only', // Specify we want object-only effect
          }),
        });
        
        console.log('Server response status:', response.status);
        console.log('Server response headers:', Object.fromEntries([...response.headers.entries()]));
        
        if (!response.ok) {
          const errorText = await response.text();
          console.error('Server error response:', errorText);
          throw new Error(`Server returned ${response.status}: ${response.statusText}\nDetails: ${errorText}`);
        }

        setDownloadingState('downloading');
        setProgress(75);

        // Download the file
        console.log('Downloading blob from response');
        const blob = await response.blob();
        console.log('Blob received, size:', blob.size, 'type:', blob.type);
        
        const fileName = getFileName('object_only');
        console.log('Saving video with filename:', fileName);
        saveVideo(blob, fileName);

        setDownloadingState('completed');
        setProgress(100);
      } catch (error: any) {
        console.error('Error downloading object video:', error);
        // Show a more user-friendly error message
        alert(`Error downloading object video: ${error.message}`);
        setDownloadingState('error');
      }

    } catch (error: any) {
      console.error('Error downloading object video:', error);
      // Show a more user-friendly error message
      alert(`Error downloading object video: ${error.message}`);
      setDownloadingState('error');
    }
  }

  function saveVideo(blob: Blob, fileName: string) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.setAttribute('href', url);
    a.setAttribute('download', fileName);
    a.setAttribute('target', '_self');
    a.click();
    window.URL.revokeObjectURL(url);
  }

  return {downloadObjectOnly, progress, state: downloadingState};
}
