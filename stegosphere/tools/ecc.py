class Hamming7_4:
    @staticmethod
    def encode(payload):
        #padding
        remainder = len(payload) % 4
        if remainder != 0:
            padding = '0' * (4 - remainder)
            payload += padding

        encoded_bits = []
        for i in range(0, len(payload), 4):
            d = [int(bit) for bit in payload[i:i+4]]
            p1 = d[0] ^ d[1] ^ d[3]
            p2 = d[0] ^ d[2] ^ d[3]
            p3 = d[1] ^ d[2] ^ d[3]
            codeword = [p1, p2, d[0], p3, d[1], d[2], d[3]]
            encoded_bits.extend(str(bit) for bit in codeword)
        return ''.join(encoded_bits)
    
    @staticmethod
    def decode(payload):
        if len(payload) % 7 != 0:
            raise ValueError("Encoded string length must be a multiple of 7.")

        decoded_bits = []
        for i in range(0, len(payload), 7):
            block = [int(bit) for bit in payload[i:i+7]]
            s1 = block[0] ^ block[2] ^ block[4] ^ block[6]
            s2 = block[1] ^ block[2] ^ block[5] ^ block[6]
            s3 = block[3] ^ block[4] ^ block[5] ^ block[6]

            syndrome = s1 + (s2 << 1) + (s3 << 2)
            if syndrome != 0:
                #error correction
                pos = syndrome - 1
                block[pos] = 1 - block[pos]
            data_bits = [block[2], block[4], block[5], block[6]]
            decoded_bits.extend(str(bit) for bit in data_bits)
        return ''.join(decoded_bits)
