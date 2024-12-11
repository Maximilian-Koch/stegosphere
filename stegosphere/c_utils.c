#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>


void get_pairs(int *shape, int pos_dim, int num_pairs, int *pairs)
{
    // Calculate the total number of indices
    int total_indices = 1;
    for (int i = 0; i < pos_dim; i++)
        total_indices *= shape[i];

    if (pos_dim > 16)
    {
        fprintf(stderr, "pos_dim exceeds maximum allowed dimension (16).\n");
        return;
    }

    int index[16] = {0}; // Assumes pos_dim <= 16

    int index_count = 0;
    int pair_index = 0;

    while (index_count < total_indices)
    {
        // Calculate the offset in the pairs array
        int offset = pair_index * 2 * pos_dim;
        int idx_in_pair = (index_count % 2) * pos_dim; // 0 or pos_dim

        // Copy the current index into the pairs array
        for (int i = 0; i < pos_dim; i++)
            pairs[offset + idx_in_pair + i] = index[i];

        // Increment pair index after storing two indices
        if (index_count % 2 == 1)
            pair_index++;

        // Advance the index array to the next index
        for (int dim = pos_dim - 1; dim >= 0; dim--)
        {
            index[dim]++;
            if (index[dim] < shape[dim])
                break;
            else
                index[dim] = 0;
        }

        index_count++;
    }
}





// Function to convert binary string to an integer array
void parse_binary_message(const char* message, int bits, int* output_array, int message_len) {
    int i, j;
    for (i = 0; i < message_len; i += bits) {
        int value = 0;
        // Convert each chunk of 'bits' size into an integer
        for (j = 0; j < bits; ++j) {
            value = (value << 1) | (message[i + j] - '0');  // Shift and add binary digit
        }
        output_array[i / bits] = value;
    }
}


/**
 * @brief Generates a binary message string from an array of integers by taking the last n binary bits.
 *      
 * @param extracted_bits Pointer to an array of int32_t, where each element is treated as 
 *                       an integer containing the bits to extract.
 * @param length The number of elements in the `extracted_bits` array.
 * @param bits The number of bits to extract from each integer in the array.
 * @param output A pointer to the output buffer where the resulting binary string is stored. 
 *               The caller must ensure that the buffer has enough space to store 
 *               (length * bits + 1) characters. The extra 1 is for the null terminator.
 *
 * @note The function writes a null-terminated string to the `output` buffer.
 *       Python needs to allocate the memory for the buffer.
 */
void generate_message_bits(int32_t* extracted_bits, int length, int bits, char* output) {
    char* p = output;
    
    for (int i = 0; i < length; i++) {
        int32_t value = extracted_bits[i];

        for (int b = bits - 1; b >= 0; b--) {
            *p++ = ((value >> b) & 1) ? '1' : '0';
        }
    }

    *p = '\0';
}
