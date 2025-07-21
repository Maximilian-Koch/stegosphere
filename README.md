![stegosphere logo](https://github.com/Maximilian-Koch/stegosphere/blob/main/logo_small_stego.png "Stegosphere Logo")
# Stegosphere

Stegosphere is a versatile library for both
applying steganography (hiding data within media) and
performing steganalysis (detecting hidden data).
It provides a flexible framework for any data representable
as a NumPy array, including images, videos, audios and TrueType fonts.
Furthermore, multi-file embedding, Hamming codes, payload compression, encryption and visualization are available, as well as a research toolbox for evaluating method performance and security.

# Documentation
Online documentation is available at https://maximilian-koch.github.io/stegosphere/

Detailed examples are available at https://github.com/Maximilian-Koch/stegosphere/tree/main/examples

# Installation
The latest release (and required dependencies) can be installed from PyPI:
`pip install stegosphere`

To install all optional dependencies for various additional operations, use:
`pip install stegosphere[complete]`

The only requirements are [numpy](https://pypi.org/project/numpy/) and [lazy-loader](https://pypi.org/project/lazy-loader/).
The file containers work with [PIL/Pillow](https://github.com/python-pillow/Pillow) for images, [fontTools](https://github.com/fonttools/fonttools) for TTF files,
[cv2](https://github.com/opencv/opencv-python) for videos, and [soundfile](https://pypi.org/project/soundfile/) for audios.
Using files containers is not required as they only provide the binding between files and numpy arrays.


# Examples

### Extracting a payload
The following example reads a hidden payload from the library's logo:
```python
from stegosphere.embeddings import LSB
from stegosphere import image
from stegosphere import binary_to_data

img = image.ImageContainer('logo_small_stego.png')
px = img.read()
payload = LSB.extract(px)

print(binary_to_data(payload).decode())
```

### Embedding and extracting
The following example hides and reads a payload from an audio file using echo hiding steganography:
```python
from stegosphere.embeddings import echo_hiding
from stegosphere import audio

aud = audio.AudioContainer('audio.wav')
frames = aud.read()
payload = '01110100011001010111001101110100'
stego_frames = echo_hiding.embed(frames,payload,alpha=0.4)
#aud.save('stego_audio.wav', stego_frames)

dec_payload = echo_hiding.extract(stego_frames,n_bits=len(payload),alpha=0.4)
print(dec_payload)
print(payload)
#minimal differences are possible due to imperfect delay detection!
```

### Advanced functionalities
The following example uses (Pixel) Value Differencing to hide a payload within an image.
The payload uses Hamming 7-4 error correcting codes and is embedded randomly across the image with a seed.
PSNR is measured and a Gaussian noise attack is being simulated.
The image is saved, loaded, the bit-error-rate is measured, and differences are visualized.
(Note: Storing and saving into a file is not needed for analysis. This is purely for demonstration.)
```python
import stegosphere
from stegosphere.embeddings import VD
from stegosphere import image
from stegosphere.costs import HILL
from stegosphere.selector import greedy_indices
from stegosphere.tools import ecc
from stegosphere.analysis import imperceptibility, robustness, accuracy, visualizer

img = image.ImageContainer('logo_small.png')
px = img.read()
payload = "payload!"*300

#convert to binary so that ECC can be applied
bin_payload = stegosphere.data_to_binary(payload)
ecc_payload = ecc.Hamming7_4.encode(bin_payload)
#do payload framing / end-of-payload manually
stego_px = VD.embed(px, ecc_payload, framing=None, seed=0)

psnr = imperceptibility.psnr(px, stego_px)
#simulating attack with very low sigma
stego_px = robustness.gaussian_noise_attack(stego_px,sigma=0.1)

print(f"PSNR: {psnr:.2f}dB")
img.save('stego_image.png',stego_px)

stego_img = image.ImageContainer('stego_image.png')
stego_px = img.read()
output = VD.extract(stego_px, framing=None, seed=0)
decc_payload = ecc.Hamming7_4.decode(output[:len(ecc_payload)])
ber = accuracy.bit_error_rate(decc_payload, bin_payload)
print(f"Bit error rate: {ber:.2f}%") #should be 0%
#Due to the seed, the differences are randomly distributed.
visualizer.difference_heatmap(px,stego_px)

```

# Contributing
Any support or input is always welcomed.
Additional embedding and cost functions are very much needed.

Contact:

Email: maximilian.koch@student.uva.nl

LinkedIn: https://www.linkedin.com/in/maximilian-jw-koch/
