import warnings
import os

import numpy as np
try:
    from fontTools.ttLib import TTFont, newTable
except:
    warnings.warn("Importing fontTools failed. font input needs to be fontTools.TTFont or numpy.array.")

from utils import *
from spatial import BaseLSB, BaseVD
from transform import BaseIWT
import core

__all__ = ['CustomTable', 'LSB']

class TTF(core.Container):
    def __init__(self, font, load_glyphs=True):
        super().__init__(font)
        #some subclasses do not require the glyphs
        if load_glyphs:
            self.data = self._get_coordinates()
        
    def _get_coordinates(self):
        glyph_data = []
        glyf_table = self.font['glyf']
        glyph_set = self.font.getGlyphSet()
        for glyph_name in glyph_set.keys():
            glyph = glyf_table[glyph_name]
            if hasattr(glyph, 'coordinates') and glyph.coordinates is not None:
                coordinates = glyph.coordinates
                glyph_data.extend(np.array(coordinates).flatten())
        return np.array(glyph_data, dtype=np.int32)
    def flush(self):
        if self.data is not None:
            glyf_table = self.font['glyf']
            index = 0
            for glyph_name in self.font.getGlyphSet().keys():
                glyph = glyf_table[glyph_name]
                if not hasattr(glyph, 'coordinates') or glyph.coordinates is None:
                    # Skip composite glyphs and glyphs without coordinates
                    continue
                coordinates = glyph.coordinates
                for i in range(len(coordinates)):
                    coordinates[i] = (self.data[index], self.data[index + 1])
                    index += 2
                glyph.coordinates = coordinates
    
    def _read_file(self, path):
        self.font = TTFont(path)
    def _flush_file(self):
        self.flush()
    def _save_file(self, path):
        self.font.save(path)

class LSB(TTF, BaseLSB):
    def __init__(self, font):
        TTF.__init__(self, font, True)
        BaseLSB.__init__(self, self.data)

class CustomTable(TTF):
    def __init__(self, font):
        TTF.__init__(self, font, False)
    def encode(self, message, table_name):
        if is_binary(message):
            warnings.warn('message does not need to be binary',
                             UserWarning)
        if not isinstance(message, bytes):
            message = message.encode('utf-8')
            warnings.warn('automatically encoded message to bytes object')
        if table_name in self.font.keys():
            warnings.warn('table name already exists. Data will be overwritten.')
        assert len(table_name)==4, Exception('table tag must be 4 characters')
        custom_table = newTable(table_name)
        custom_table.data = message
        self.font[table_name] = custom_table
        return True
    def decode(self, table_name):
        assert len(table_name)==4, Exception('table tag must be 4 characters')
        if table_name in self.font:
            return self.font[table_name].data
        else:
            #table name not in tables
            return None
