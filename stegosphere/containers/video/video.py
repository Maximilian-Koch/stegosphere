"""
An experimental module for video containers.
"""
import os
import numpy as np

from stegosphere.containers import container

try:
    import cv2
except ImportError as e:
    raise ImportError("VideoContainer requires OpenCV (cv2). Install via `pip install opencv-python`.") from e



class VideoContainer(container.Container):
    """
    File container for video files.
    """
    def __init__(self, video):
        if hasattr(cv2, 'VideoCapture') and isinstance(video, cv2.VideoCapture):
            self.cap = video
            self._owns_capture = False
        elif isinstance(video, str):
            if not os.path.isfile(video):
                raise FileNotFoundError(f"No such video file: '{video}'")
            self.cap = cv2.VideoCapture(video)
            self._owns_capture = True
            if not self.cap.isOpened():
                self.cap.release()
                raise OSError(f"Cannot open video file '{video}'")
        else:
            raise TypeError("VideoContainer requires a cv2.VideoCapture or valid file path")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames = None

    def read(self):
        """
        Reads all frames from the video
        """
        frames = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        self.cap.release()
        return np.array(frames)

    def flush(self, frames):
        """
        Flushes the modified frames
        """
        if not isinstance(frames, (list, np.ndarray)):
            raise ValueError("Frames must be a list or numpy array.")
        self.frames = frames

    def save(self, path, flush=None, codec='FFV1'):
        """
        Saves the flushed frames to path
        """
        if flush is not None:
            self.flush(flush)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        height, width, _ = self.frames[0].shape

        out = cv2.VideoWriter(path, fourcc, self.fps, (width, height))

        for frame in self.frames:
            out.write(frame)

        out.release()
