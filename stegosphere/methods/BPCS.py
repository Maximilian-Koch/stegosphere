#This implements Bit Plane Complexity Segmentation as based from:
#Kawaguchi, E., & Eason, R. O. (1999, January).
#Principles and applications of BPCS steganography.
#In Multimedia systems and applications (Vol. 3528, pp. 464-473). SPIE.


import numpy as np


def embed(array, payload, block_size = 8, threshold = 0.3):
    """
    Generalized BPCS embedding.

    :param array: array to be embedded into
    :type array: np.ndarray
    :param payload: payload
    :type payload: str
    
    Returns (array, conj_map_records, used_bits).
    """
    assert array.ndim in [2,3], 'Array must be 2 or 3 dimensional'
    
    if array.ndim == 2:
        array_3d = array[:, :, np.newaxis]
    else:
        array_3d = array

    H, W, C = array_3d.shape
    array_chw = np.transpose(array_3d, (2, 0, 1))  # shape: (C,H,W)

    payload_index = 0
    payload_length = len(payload)
    conj_map_records = []

    for channel_idx in range(C):
        if payload_index >= payload_length:
            break

        channel_data = array_chw[channel_idx]
        bitplanes = _split_into_bitplanes(channel_data)

        # Iterate over bitplanes
        for bitplane_idx in range(8):
            if payload_index >= payload_length:
                break

            bp = bitplanes[bitplane_idx]

            # Segment into blocks
            for by in range(0, H, block_size):
                for bx in range(0, W, block_size):
                    if payload_index >= payload_length:
                        break

                    block = bp[by:by+block_size, bx:bx+block_size]
                    if block.shape != (block_size, block_size):
                        continue  # skip partial blocks

                    # Complexity check
                    c = _compute_complexity(block)
                    if c >= threshold:
                        max_bits = block_size * block_size
                        bits_left = payload_length - payload_index
                        bits_to_embed = min(max_bits, bits_left)

                        data_chunk = payload[payload_index : payload_index + bits_to_embed]


                        data_arr = np.zeros((block_size, block_size), dtype=np.int32)
                        flat = data_arr.ravel()
                        for i in range(bits_to_embed):
                            flat[i] = int(data_chunk[i])
                        data_arr = flat.reshape((block_size, block_size))

                        # Check complexity of data_arr
                        data_complexity = _compute_complexity(data_arr)
                        was_conjugated = False
                        if data_complexity < threshold:
                            data_arr = 1 - data_arr
                            was_conjugated = True

                        # Embed
                        block[:, :] = data_arr
                        payload_index += bits_to_embed

                        # Store record
                        conj_map_records.append((channel_idx, bitplane_idx, by, bx, was_conjugated))

            # Reconstruct bitplane after changes
            bitplanes[bitplane_idx] = bp

        # Rebuild the channel from the modified bitplanes
        array_chw[channel_idx] = _reconstruct_from_bitplanes(bitplanes)

    used_bits = payload_index

    # Re-shape back (H,W,C)
    array_3d = np.transpose(array_chw, (1, 2, 0))
    if array_3d.shape[2] == 1:
        # squeeze to (H,W) if grayscale
        array_3d = np.squeeze(array_3d, axis=2)

    return array_3d.astype(np.uint8), conj_map_records, used_bits

def extract(array,conj_map_records,total_bits,block_size = 8):
    """
    Generalized BPCS extraction. 
    conj_map_records = [(channel_idx, bitplane_idx, by, bx, was_conjugated), ...]
    Returns binary string of length total_bits.
    """
    #make 3d
    if array.ndim == 2:
        array_3d = array[:, :, np.newaxis]
    else:
        array_3d = array

    H, W, C = array_3d.shape
    array_chw = np.transpose(array_3d, (2, 0, 1))

    extracted_bits = []
    bits_collected = 0

    for (channel_idx, bitplane_idx, by, bx, was_conjugated) in conj_map_records:
        if bits_collected >= total_bits:
            break

        channel_data = array_chw[channel_idx]
        bitplanes = _split_into_bitplanes(channel_data)
        bp = bitplanes[bitplane_idx]

        block = bp[by:by+block_size, bx:bx+block_size]
        if block.shape != (block_size, block_size):
            continue

        flat_block = block.ravel()
        if was_conjugated:
            flat_block = 1 - flat_block

        max_block_bits = block_size * block_size
        needed_bits = min(max_block_bits, total_bits - bits_collected)

        extracted_bits.extend(flat_block[:needed_bits].tolist())
        bits_collected += needed_bits

    extracted_bin = ''.join(str(bit) for bit in extracted_bits)
    return extracted_bin[:total_bits]


def _split_into_bitplanes(channel_data: np.ndarray):
    """
    Split a single-channel 8-bit image (H,W) into a list of 8 bitplanes [0..7].
    bitplane[0] = LSB, bitplane[7] = MSB.
    """
    bitplanes = []
    for b in range(8):
        bp = (channel_data >> b) & 1
        bitplanes.append(bp.astype(np.int32))
    return bitplanes


def _reconstruct_from_bitplanes(bitplanes):
    """
    Reconstruct a single-channel 8-bit image from a list of 8 bitplanes.
    bitplane[0] = LSB, bitplane[7] = MSB.
    """
    channel_data = np.zeros_like(bitplanes[0], dtype=np.uint8)
    for b in range(8):
        channel_data |= (bitplanes[b].astype(np.uint8) << b)
    return channel_data


def _compute_complexity(block: np.ndarray):
    """
    Simple complexity measure: count 0->1 or 1->0 transitions in horizontal + vertical directions.
    Returns a float in [0,1].
    """
    h, w = block.shape
    transitions = 0

    # Horizontal transitions
    for row in range(h):
        for col in range(w - 1):
            if block[row, col] != block[row, col + 1]:
                transitions += 1

    # Vertical transitions
    for col in range(w):
        for row in range(h - 1):
            if block[row, col] != block[row + 1, col]:
                transitions += 1

    max_transitions = (h * (w - 1)) + (w * (h - 1))
    return transitions / max_transitions if max_transitions > 0 else 0
