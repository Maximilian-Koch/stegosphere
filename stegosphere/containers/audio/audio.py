"""
Defines containers for audio files.
"""

import os
import numpy as np
import soundfile as sf

from stegosphere.containers import container

class AudioContainer(container.Container):
    """
    General audio file container using pysoundfile (libsndfile).
    Supports reading, flushing new frames, and saving while preserving metadata.
    """
    def __init__(self, audio):
        if isinstance(audio, sf.SoundFile):
            self._sf = audio
            self._owns_handle = False
        elif isinstance(audio, str) and os.path.isfile(audio):
            try:
                self._sf = sf.SoundFile(audio, mode='r')
            except RuntimeError as e:
                raise OSError(f"Cannot open audio file '{audio}': {e}") from e
            self._owns_handle = True
        else:
            raise TypeError("AudioContainer requires a soundfile.SoundFile or existing file path")

        #metadata
        self.samplerate = self._sf.samplerate
        self.channels   = self._sf.channels
        self.subtype    = self._sf.subtype
        self.format     = self._sf.format
        self.frames     = None

    def read(self):
        """
        Read frames
        """
        self._sf.seek(0)
        data = self._sf.read(always_2d=True)
        self._sf.seek(0)
        return data

    def flush(self, frames):
        """
        Buffer a new frame-array for later saving.

        Args:
            frames (numpy.ndarray): Frames
        Returns:
            None
        Raises:
            ValueError: if provided frames is not in the correct format
        """
        if frames.ndim == 1:
            if self.channels == 1:
                frames = frames[:, None]
            else:
                raise ValueError(f"flush() 1D array only allowed for mono audio, got {self.channels} channels")
        
        if frames.ndim != 2 or frames.shape[1] != self.channels:
            raise ValueError(f"flush() array must be 2D with {self.channels} channels")
        self.frames = frames

    def save(self, path, flush = None, keep_meta = True):
        """
        Write audio to `path`. If `flush` is provided, buffers it first.

        Args:
            path (str): output filepath
            flush (numpy.ndarray, optional): optional 2D numpy array to write instead of original
            keep_meta (bool, optional): whether to preserve samplerate/subtype/format
        """
        if flush is not None:
            self.flush(flush)

        if self.frames is None:
            data = self.read()
        else:
            data = self.frames

        sr    = self.samplerate
        fmt   = self.format    if keep_meta else None
        sub   = self.subtype   if keep_meta else None

        try:
            sf.write(file=path,
                     data=data,
                     samplerate=sr,
                     format=fmt,
                     subtype=sub)
        except RuntimeError as e:
            raise OSError(f"Failed to write audio file '{path}': {e}") from e
        finally:
            if getattr(self, "_owns_handle", False):
                self._sf.close()
