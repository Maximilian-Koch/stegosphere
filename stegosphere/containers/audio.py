import wave
import os

import numpy as np

from stegosphere import base

class WAVContainer(base.Container):
    """
    File container for wave files, for reading, writing and saving.
    """
    def __init__(self, audio):
        if isinstance(audio, wave.Wave_read):
            self.audio = audio
        elif os.path.isfile(audio):
            self.audio = wave.open(audio, 'rb')
        elif isinstance(audio, np.ndarray):
            raise Exception('No need for containers for arrays')

        else:
            raise Exception('wav must be wave.Wave_read or path')

        self.meta = self.audio.getparams()
        self.frames = None
    def read(self):
        """
        Read frames.
        """
        frame_bytes = self.audio.readframes(self.audio.getnframes())
        sample_width = self.audio.getsampwidth()
        dtype = np.int16 if sample_width == 2 else np.uint8  # Adjust for bit depth
        samples = np.frombuffer(frame_bytes, dtype=dtype)
        self.audio.setpos(0)
        return samples.reshape(-1, self.meta.nchannels)

    def flush(self, frames):
        """
        Flush new frames into container.

        :param frames:
        :type frames: numpy.ndarray
        """
        self.frames = frames.flatten().tobytes()

    def save(self, path, keep_meta=True):
        """
        Save audio to path.

        :param path: Path to new file
        :param keep_meta: Keep basic information from initial file
        """
        assert self.frames is not None, 'Flush frames first'
        with wave.open(path, 'wb') as output:
            if keep_meta:
                output.setparams(self.meta)
            output.writeframes(self.frames)
