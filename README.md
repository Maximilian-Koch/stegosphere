# Stegosphere
A flexible, highly modular steganography library for all NumPy-convertible file types including encryption and compression, with an optional C backend.
An initial documentation is available here: https://maximilian-koch.github.io/stegosphere/stegosphere.html#submodules


### Image steganography
For image steganography, LSB (Least Significant Bit) and PVD (Pixel Value Differencing) steganography are currently available.
```
import image

img = image.LSB('image.png')
img.encode('Encoded message!', seed=42, method='delimiter', compress=False)
img.save('stego_image.png')

steg_img = image.LSB('stego_image.png')
print(steg_img.decode(seed=42, method='delimiter', compress=False))
#Expected output: 'Encoded message!'
```
### Audio steganography
For audio steganography, LSB (Least Significant Bit) and FVD (Frequency Value Differencing) steganography are currently available.
```
import audio

audio = audio.FVD('audio.wav')
bin_image = audio.file_to_binary('image.png')
audio.encode(bin_image)
audio.save('steg_audio.wav')

steg_audio = audio.LSB('steg_audio.wav')
audio.binary_to_file(steg_audio.decode(), 'recovered_image.png')
```
