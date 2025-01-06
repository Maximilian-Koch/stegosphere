from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import binascii

class AES_GCM:
    @staticmethod
    def encrypt(payload, key, iv=None):
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')
        if iv is None:
            iv = os.urandom(12)
        assert len(iv) == 12, "IV must be 12 bytes for AES-GCM"
        
        encryptor = Cipher(algorithms.AES(key), modes.GCM(iv)).encryptor()
        ciphertext = encryptor.update(payload) + encryptor.finalize()
        encrypted_data = iv + ciphertext + encryptor.tag
        return binascii.hexlify(encrypted_data).decode('utf-8')

    @staticmethod
    def decrypt(ciphertext, key):
        if isinstance(ciphertext, str):
            ciphertext = binascii.unhexlify(ciphertext)
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        iv = ciphertext[:12]
        tag = ciphertext[-16:]
        actual_ciphertext = ciphertext[12:-16]
        
        decryptor = Cipher(algorithms.AES(key), modes.GCM(iv, tag)).decryptor()
        plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()
        return plaintext.decode('utf-8')
