import os

import numpy as np
try:
    from fontTools.ttLib import TTFont, newTable
except:
    raise Exception("Importing fontTools failed. font input needs to be fontTools.TTFont or numpy.array.")


from stegosphere import base

__all__ = ['TTFContainer']

class TTFContainer(base.Container):
    """
    File container for TrueType Font files.
    """
    def __init__(self, font):
        if isinstance(font, TTFont):
            self.font = font
        elif os.path.isfile(font):
            self.font = TTFont(font)
        elif isinstance(font, np.ndarray):
            raise Exception('No need for containers for arrays')
        
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
    def save(self, path):
        """
        Save font
        """
        self.font.save(path)
