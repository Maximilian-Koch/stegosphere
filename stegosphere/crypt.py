from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os


class AES_GCM:
    @staticmethod
    def encrypt(message, key, iv = None):
        if isinstance(message, str): message = bytes(message, 'utf-8')
        if isinstance(key, str): key = bytes(key, 'utf-8')
        if iv == None:
            iv = os.urandom(12)
        assert len(iv) == 12
        encryptor = Cipher(algorithms.AES(key), modes.GCM(iv)).encryptor()
        ciphertext = encryptor.update(message) + encryptor.finalize()
        return (iv + ciphertext + encryptor.tag).hex()
    @staticmethod
    def decrypt(message, key):
        iv, ciphertext, tag = ciphertext[:12], ciphertext[12:-16], ciphertext[-16:]
        decryptor = Cipher(algorithms.AES(key), modes.GCM(iv, tag)).decryptor()
        return (decryptor.update(ciphertext) + decryptor.finalize()).decode()

