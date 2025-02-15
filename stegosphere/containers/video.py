import os

import numpy as np
try:
    import cv2
except:
    raise Exception('VideoContainer requires cv2.')

from stegosphere.containers import container


class VideoContainer(container.Container):
    """
    File container for video files.
    """
    def __init__(self, video):
        if isinstance(video, cv2.VideoCapture):
            self.cap = video
        elif os.path.isfile(video):
            self.cap = cv2.VideoCapture(video)
            if not self.cap.isOpened():
                raise Exception("Could not open the video file.")

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

    def save(self, path, codec='FFV1'):
        """
        Saves the flushed frames to path
        """
        assert self.frames is not None, "No frames to save. Flush frames first."
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        height, width, _ = self.frames[0].shape

        out = cv2.VideoWriter(path, fourcc, self.fps, (width, height))

        for frame in self.frames:
            out.write(frame)

        out.release()
