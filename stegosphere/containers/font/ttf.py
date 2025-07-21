"""
Module for interacting with ttf files.
"""

import os
import numpy as np

from stegosphere.containers import container

try:
    from fontTools.ttLib import TTFont
except ImportError as e:
    raise ImportError("TTFContainer requires fontTools. Install via `pip install fonttools`.") from e


__all__ = ['TTFContainer']

class TTFContainer(container.Container):
    """
    File container for TrueType Font files.
    """
    def __init__(self, font):
        if isinstance(font, TTFont):
            self.font = font
            self._owns_font = False
        elif isinstance(font, str):
            if not os.path.isfile(font):
                raise FileNotFoundError(f"No such font file: '{font}'")
            try:
                self.font = TTFont(font)
            except Exception as e:
                raise OSError(f"Failed to open TTF file '{font}': {e}") from e
            self._owns_font = True
        else:
            raise TypeError("TTFContainer requires a TTFont or valid file path")
        
    def read(self):
        """
        Read glyphs.
        """
        glyph_data = []
        glyf_table = self.font['glyf']
        glyph_set = self.font.getGlyphSet()
        for glyph_name in glyph_set.keys():
            glyph = glyf_table[glyph_name]
            if hasattr(glyph, 'coordinates') and glyph.coordinates is not None:
                coordinates = glyph.coordinates
                glyph_data.extend(np.array(coordinates).flatten())
        return np.array(glyph_data, dtype=np.int32)
    def flush(self, glyphs):
        """
        Flush new glyphs into container.

        Args:
            glyphs (iterable): List of glyphs
        Returns:
            None
        """
        glyf_table = self.font['glyf']
        index = 0
        for glyph_name in self.font.getGlyphSet().keys():
            glyph = glyf_table[glyph_name]
            if not hasattr(glyph, 'coordinates') or glyph.coordinates is None:
                    # Skip composite glyphs and glyphs without coordinates
                continue
            coordinates = glyph.coordinates
            for i in range(len(coordinates)):
                coordinates[i] = (glyphs[index], glyphs[index + 1])
                index += 2
            glyph.coordinates = coordinates
    def save(self, path, flush=None, close_file=True):
        """
        Save font
        """
        if flush is not None:
            self.flush(flush)
        self.font.save(path)
        if getattr(self, "_owns_font", False)  and close_file:
            self.font.close()
