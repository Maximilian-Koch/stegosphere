"""
Creating/reading a custom table within a ttf file.
"""

import numpy as np
from fontTools.ttLib import TTFont, newTable


def embed(font, payload, table_name):
    """
    Create new table in ttf file.

    Args:
        font (stegosphere.font.ttf.TTFContainer): Font object
        payload (str | bytes): Payload to write into table
        table_name (str): 4-character table tag
    Returns:
        stegosphere.font.ttf.TTFContainer: Updated font object
    Raises:
        ValueError: If `table_name` is not 4 characters long
    Warns:
        UserWarning: If `table_name` already exists, thus data will be overwritten
    """
    
    if not isinstance(payload, bytes):
        payload = payload.encode('utf-8')
    if table_name in font.font.keys():
        warnings.warn('table name already exists. Data will be overwritten.', UserWarning)
    if len(table_name) != 4:
        raise ValueError('Table name tag must be 4 characters long')
    custom_table = newTable(table_name)
    custom_table.data = payload
    font.font[table_name] = custom_table
    return font

def extract(font, table_name):
    """
    Extract table from ttf file.

    Args:
        font (stegosphere.font.ttf.TTFContainer): Font object
        table_name (str): Table to read
    Returns:
        table data or None, if `table_name` not in `font`
    Raises:
        ValueError: If `table_name` is not 4 characters long
    """
    if len(table_name) != 4:
        raise ValueError('Table name tag must be 4 characters long')
    if table_name in font.font:
            return font.font[table_name].data
    else:
        #table name not in tables
        return None

