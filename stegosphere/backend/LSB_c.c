#include <stdint.h>
#include <string.h>

/**
 * Embed a binary string into an array by modifying the least significant bits of each element.
 *
 * Args:
 *   arr          - Pointer to the array to embed into.
 *   length       - Number of elements in the array.
 *   bitstring    - Null-terminated binary string to embed.
 *   bits         - Number of least significant bits (LSBs) to use per element.
 *   element_size - Size in bytes of each array element (must be 1, 2, 4, or 8).
 *   indices      - Optional array of indices to embed into. If NULL, uses sequential order.
 */

__declspec(dllexport)
void embed(
    void*    arr,
    int      length,
    const char* bitstring,
    int      bits,
    int      element_size,
    const int* indices
) {
    int total_bits = (int)strlen(bitstring);
    if (total_bits == 0 || bits <= 0) return;

    // how many elements we need to touch (ceil)
    int elems = (total_bits + bits - 1) / bits;
    if (elems > length) elems = length;

    uint64_t mask = ((uint64_t)1 << bits) - 1;

    for (int i = 0, bit_pos = 0; i < elems; ++i, bit_pos += bits) {
        int idx = indices ? indices[i] : i;
        char* ptr = (char*)arr + idx * element_size;

        // pack up to 'bits' LSBs from bitstring
        int remain = total_bits - bit_pos;
        int take   = remain < bits ? remain : bits;
        uint64_t chunk = 0;
        // gather only what's left, align to LSB
        for (int b = 0; b < take; ++b) {
            chunk = (chunk << 1) | (bitstring[bit_pos + b] == '1');
        }

        switch (element_size) {
            case 1: {
                uint8_t v = *(uint8_t*)ptr;
                v = (v & ~((uint8_t)mask)) | (uint8_t)chunk;
                *(uint8_t*)ptr = v;
                break;
            }
            case 2: {
                uint16_t v = *(uint16_t*)ptr;
                v = (v & ~((uint16_t)mask)) | (uint16_t)chunk;
                *(uint16_t*)ptr = v;
                break;
            }
            case 4: {
                uint32_t v = *(uint32_t*)ptr;
                v = (v & ~((uint32_t)mask)) | (uint32_t)chunk;
                *(uint32_t*)ptr = v;
                break;
            }
            case 8: {
                uint64_t v = *(uint64_t*)ptr;
                v = (v & ~mask) | chunk;
                *(uint64_t*)ptr = v;
                break;
            }
            default:
                // unsupported
                return;
        }
    }
}

/**
 * Extract a binary string from the least significant bits of an array.
 *
 * Args:
 *   arr          - Pointer to the input array.
 *   length       - Number of elements to read from.
 *   bits         - Number of LSBs to extract from each element.
 *   element_size - Size in bytes of each array element (must be 1, 2, 4, or 8).
 *   out_str      - Output buffer to write extracted bits as a null-terminated string.
 *   indices      - Optional array of indices to extract from. If NULL, uses sequential order.
 */
__declspec(dllexport)
void extract(
    const void* arr,
    int     length,
    int     bits,
    int     element_size,
    char*   out_str,
    const int* indices
) {
    if (bits <= 0) { out_str[0] = '\0'; return; }
    uint64_t mask = ((uint64_t)1 << bits) - 1;
    int total_bits = length * bits;

    for (int i = 0; i < length; ++i) {
        int idx = indices ? indices[i] : i;
        const char* ptr = (const char*)arr + idx * element_size;
        uint64_t v = 0;

        switch (element_size) {
            case 1: v = *(const uint8_t*)ptr; break;
            case 2: v = *(const uint16_t*)ptr; break;
            case 4: v = *(const uint32_t*)ptr; break;
            case 8: v = *(const uint64_t*)ptr; break;
            default: v = 0; 
        }
        uint64_t chunk = v & mask;
        // write bits MSB first
        for (int b = 0; b < bits; ++b) {
            int out_idx = i * bits + b;
            if (out_idx >= total_bits) break;
            out_str[out_idx] = (chunk & (1ULL << (bits - 1 - b))) ? '1' : '0';
        }
    }
    out_str[total_bits] = '\0';
}
