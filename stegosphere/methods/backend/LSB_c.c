#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
 * embed:
 *   arr          - pointer to the array of integer elements (any size: 8,16,32,64 bits)
 *   length       - number of elements in the array
 *   bitstring    - binary string to embed (e.g., "101010..."), up to param*length bits
 *   param        - number of overwritten LSBs in each element
 *   element_size - size of each element in bytes (1, 2, 4, or 8)
 *   indices      - if non-NULL, an array of indices that specifies the order in which to embed bits.
 *
 * Overwrites 'param' LSBs of each selected element in arr with bits from bitstring.
 * If bitstring is shorter than param*length, not all elements will be updated.
 * If bitstring is longer, extra bits are ignored.
 */
__declspec(dllexport)
void embed(
    void* arr,
    int length,
    const char* bitstring,
    int param,
    int element_size,
    const int* indices  // new parameter for random indices
) {
    int bitstring_len = (int)strlen(bitstring);           // total bits in bitstring
    int max_elements_for_bits = bitstring_len / param;      // how many elements can store bits fully
    int embed_count = (length < max_elements_for_bits) ? length : max_elements_for_bits;
    
    // For each element that will store bits:
    for(int i = 0; i < embed_count; i++) {
        int index = indices ? indices[i] : i;  // use random index if provided
        // Address of the element to modify
        char* elem_ptr = (char*)arr + index * element_size;
        
        // Gather param bits from the bitstring into a temporary variable 'bits'
        uint64_t bits = 0;
        for(int b = 0; b < param; b++) {
            int idx = i * param + b;  // index in the bitstring
            char c = bitstring[idx];  // '0' or '1'
            int bit_val = (c == '1') ? 1 : 0;
            bits = (bits << 1) | bit_val;
        }
        
        // Overwrite the param LSBs using masking based on the element size
        switch(element_size) {
            case 1: {
                uint8_t val = *((uint8_t*)elem_ptr);
                uint8_t mask = (1U << param) - 1U;
                val &= ~mask;
                val |= (uint8_t)(bits & mask);
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
 *   length       - number of elements to extract bits from
 *   param        - how many LSBs to read from each element
 *   element_size - size of each element in bytes
 *   out_str      - buffer to store extracted bits (as '0'/'1' chars)
 *   indices      - if non-NULL, an array of indices that specifies the order in which to extract bits.
 *
 * Reads 'param' bits from each selected element and reconstructs them into out_str.
 * out_str should be at least param*length + 1 in size to store the bits plus the null terminator.
 */
__declspec(dllexport)
void extract(
    const void* arr,
    int length,
    int param,
    int element_size,
    char* out_str,
    const int* indices  // new parameter for random indices
) {
    int total_bits = param * length;  // total bits we will extract
    
    for(int i = 0; i < length; i++) {
        int index = indices ? indices[i] : i;  // use provided random index if available
        // Address of the element to read
        const char* elem_ptr = (const char*)arr + index * element_size;
        
        uint64_t val = 0;
        switch(element_size) {
            case 1: val = *((const uint8_t*)elem_ptr); break;
            case 2: val = *((const uint16_t*)elem_ptr); break;
            case 4: val = *((const uint32_t*)elem_ptr); break;
            case 8: val = *((const uint64_t*)elem_ptr); break;
            default: break;
        }
        
        // Extract param bits from val
        uint64_t mask = ((uint64_t)1 << param) - 1ULL;
        uint64_t stored_bits = val & mask;
        
        // Write the bits as characters ('0' or '1') into out_str
        for(int b = 0; b < param; b++) {
            int out_idx = i * param + b;
            if(out_idx >= total_bits) break;
            int shift = param - 1 - b;
            int bit_val = (int)((stored_bits >> shift) & 1ULL);
            out_str[out_idx] = bit_val ? '1' : '0';
        }
    }
    
    out_str[total_bits] = '\0';  // null-terminate the string
}
