

import numpy as np
from fontTools.ttLib import TTFont, newTable


def embed(font, payload, table_name):
    """
    Crete new table in ttf file.
    """
    
    if not isinstance(payload, bytes):
        payload = payload.encode('utf-8')
    if table_name in font.font.keys():
        warnings.warn('table name already exists. Data will be overwritten.')
    assert len(table_name)==4, Exception('table tag must be 4 characters')
    custom_table = newTable(table_name)
    custom_table.data = payload
    font.font[table_name] = custom_table
    return font

def extract(font, table_name):
    """
    Extract table from ttf file.
    """
    assert len(table_name)==4, Exception('table tag must be 4 characters')
    if table_name in font.font:
            return font.font[table_name].data
    else:
        #table name not in tables
        return None

