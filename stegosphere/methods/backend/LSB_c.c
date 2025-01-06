#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
 * embed:
 *   arr          - pointer to the array of integer elements (any size: 8,16,32,64 bits)
 *   length       - number of elements in the array
 *   bitstring    - binary string to embed (e.g., "101010..."), up to param*length bits
 *   param        - how many LSBs we overwrite in each element
 *   element_size - size of each element in bytes (1, 2, 4, or 8)
 *
 * Overwrites 'param' LSBs of each element in arr with bits from bitstring.
 * If bitstring is shorter than param*length, not all elements will be updated.
 * If bitstring is longer, extra bits are ignored.
 */
__declspec(dllexport)
void embed(
    void* arr,
    int length,
    const char* bitstring,
    int param,
    int element_size
) {
    int bitstring_len = (int)strlen(bitstring);           // total bits in bitstring
    int max_elements_for_bits = bitstring_len / param;    // how many elements can store bits fully
    int embed_count = (length < max_elements_for_bits) ? length : max_elements_for_bits;
    
    // For each element that will store bits:
    for(int i = 0; i < embed_count; i++) {
        // Gather param bits from the bitstring into a temporary variable 'bits'
        // e.g. if param=3 and bitstring segment is "101", bits = 0b101 (decimal 5)
        uint64_t bits = 0;
        for(int b = 0; b < param; b++) {
            int idx = i * param + b;  // index in the bitstring
            char c = bitstring[idx];  // '0' or '1'
            int bit_val = (c == '1') ? 1 : 0;
            // shift left, add new bit (MSB-first approach)
            bits = (bits << 1) | bit_val;
        }
        
        // Address of the i-th element
        char* elem_ptr = (char*)arr + i * element_size;
        
        // We'll interpret the element as an unsigned integer to do bit masking
        switch(element_size) {
            case 1: {
                uint8_t val = *((uint8_t*)elem_ptr);
                uint8_t mask = (1U << param) - 1U;  // e.g. param=2 -> 0b11
                val &= ~mask;                      // zero out the param LSBs
                val |= (uint8_t)(bits & mask);     // set param LSBs
                *((uint8_t*)elem_ptr) = val;
                break;
            }
            case 2: {
                uint16_t val = *((uint16_t*)elem_ptr);
                uint16_t mask = (1U << param) - 1U;
                val &= ~mask;
                val |= (uint16_t)(bits & mask);
                *((uint16_t*)elem_ptr) = val;
                break;
            }
            case 4: {
                uint32_t val = *((uint32_t*)elem_ptr);
                uint32_t mask = (1U << param) - 1U;
                val &= ~mask;
                val |= (uint32_t)(bits & mask);
                *((uint32_t*)elem_ptr) = val;
                break;
            }
            case 8: {
                uint64_t val = *((uint64_t*)elem_ptr);
                uint64_t mask = ((uint64_t)1 << param) - 1ULL;
                val &= ~mask;
                val |= (bits & mask);
                *((uint64_t*)elem_ptr) = val;
                break;
            }
            default:
                // Unsupported size
                break;
        }
    }
}

/*
 * extract:
 *   arr          - pointer to the array of integer elements
 *   length       - number of elements in the array
 *   param        - how many LSBs to read from each element
 *   element_size - size of each element in bytes
 *   out_str      - buffer to store extracted bits (as '0'/'1' chars)
 *
 * Reads 'param' bits from each element and reconstructs them into out_str,
 * in the same order they were embedded. out_str should be at least
 * param*length + 1 in size, to store the bits plus null terminator.
 */
__declspec(dllexport)
void extract(
    const void* arr,
    int length,
    int param,
    int element_size,
    char* out_str
) {
    int total_bits = param * length;  // total bits we can extract
    
    for(int i = 0; i < length; i++) {
        // Address of the i-th element
        const char* elem_ptr = (const char*)arr + i * element_size;
        
        // We'll interpret the element as an unsigned integer
        uint64_t val = 0;
        switch(element_size) {
            case 1: val = *((const uint8_t*)elem_ptr); break;
            case 2: val = *((const uint16_t*)elem_ptr); break;
            case 4: val = *((const uint32_t*)elem_ptr); break;
            case 8: val = *((const uint64_t*)elem_ptr); break;
            default: break;
        }
        
        // Extract param bits from val
        // e.g. if param=3 and val's LSBs are 0b101, we want to produce "101"
        // In embed_lsb_any, we used (bits << 1) for each bit, so the leftmost bit is the MSB among param bits.
        uint64_t mask = ((uint64_t)1 << param) - 1ULL;
        uint64_t stored_bits = val & mask;
        
        // Reconstruct these bits as '0' or '1'
        for(int b = 0; b < param; b++) {
            int out_idx = i * param + b;
            // Safety check
            if(out_idx >= total_bits) break;
            
            // The bit we want is from stored_bits' MSB down to LSB
            // Example: if stored_bits=0b101 (decimal 5) with param=3,
            //   b=0 -> we want MSB -> (shift=2) -> 1
            //   b=1 -> shift=1 -> 0
            //   b=2 -> shift=0 -> 1
            int shift = param - 1 - b;
            int bit_val = (int)((stored_bits >> shift) & 1ULL);
            out_str[out_idx] = bit_val ? '1' : '0';
        }
    }
    
    // Null-terminate
    out_str[total_bits] = '\0';
}
