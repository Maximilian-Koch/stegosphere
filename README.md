# Stegosphere
Stegosphere is a versatile library for both
applying steganography (hiding data within media) and
performing steganalysis (detecting hidden data).
It provides a flexible framework for any data representable
as a NumPy array, including images, videos, audio and TTF fonts.
Furthermore, multi-file embedding, Hamming codes, payload compression and encryption are available, as well as a research toolbox for evaluating method performance and security, see [Research toolbox](#research-toolbox).


# Table of contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Image steganography](#image-steganography)
4. [Audio steganography](#audio-steganography)
5. [ttf steganography](#ttf-steganography)
6. [wavelet transform](#wavelet-transform)
7. [Multifile steganography](#multifile-steganography)
8. [File handling](#file-handling)
9. [Compression and Encryption](#compression-and-encryption)
10. [Research toolbox](#research-toolbox)
11. [Contributing](#contributing)

## Overview
**Core Design: NumPy-Centric Approach**

Most algorithms within the library operate directly on Numpy arrays,
thus the same fundamental techniques (such as LSB and Value Differencing) can be applied to any data source,
and data can be processed before and after as needed.

**File Handling via Containers**

To bridge the gap between specific file formats and the 
NumPy-based methods, the `stegosphere.containers` module can be used.
Currently, images, video, wav and ttf files containers are available.
However, the translation into a NumPy array can
also be done manually by the user.
The container classes handle reading data into 
an appropriate array, flushing the data and saving back to a file.
The image container works with [PIL/Pillow](https://github.com/python-pillow/Pillow), so that the original metadata
stays the same, which is more complicated when using cv2.
The WAV container uses the standard wave library,
TTF font uses [fontTools](https://github.com/fonttools/fonttools) and the video container with [cv2](https://github.com/opencv/opencv-python).

**Steganographic Methods**

The pre-processing, embedding and extraction methods are within the `stegosphere.methods` module.
Currently the following methods are implemented:
* Least Significant Bit (LSB)
* Value Differencing (PVD for images, FVD for audio)
* Bit-Plane Complexity Segmentation (BPCS)
* Integer Wavelet Transform (IWT)
* Custom Table creation (TTF files only)

## Installation
Install using pip: `pip install stegosphere`.

The only requirement is `numpy`.
The file containers work with `PIL` for images, `fontTools` for ttf and `cv2` for videos, however the file containers are not required to be used as they only provide the binding between files and numpy arrays.

## Image steganography
For image steganography, LSB (Least Significant Bit), PVD (Pixel Value Differencing), BPCS (Bit-Plane Complexity Segmentation) and IWT (Integer Wavelet Transform) steganography are currently available.

The example loads an image into a container and writes a payload into the pixels and then reads it out again.
```python
from stegosphere.containers.image import ImageContainer
from stegosphere.methods import LSB
from stegosphere.io import *

img = ImageContainer('image.png')
px = img.read()
cover = LSB.embed(px, 'Embedded message!', method='delimiter')
img.flush(cover)
img.save('image_stego.png')

steg_img = ImageContainer('image_stego.png')
steg_px = steg_img.read()
uncover = LSB.extract(steg_px, method='delimiter')

print(binary_to_data(uncover))
#Expected output: 'Embedded message!'
```
## Audio steganography
For audio steganography, LSB (Least Significant Bit), FVD (Frequency Value Differencing) and IWT (Integer Wavelet Transform) steganography are currently available.
The example below loads an audio and encodes the file `image.png` into the audio. The image is then recovered and saved.
```python
from stegosphere.containers.audio import WAVContainer
from stegosphere.methods import VD
from stegosphere.io import file_to_binary, binary_to_file


wav = WAVContainer('audio.wav')
bin_image = file_to_binary('image.png')
frames = wav.read()
embedded = VD.embed(frames, bin_image)
wav.flush(embedded)
wav.save('steg_audio.wav')


steg_wav = WAVContainer('steg_audio.wav')
steg_frames = steg_wav.read()
extracted = VD.extract(steg_frames)

binary_to_file(extracted,'recovered_image.png')

```
For additional parameters, see the chapter on [parameters](#additional-parameters).

## ttf steganography
For ttf steganography, LSB (Least Significant Bit) and Custom Table steganography are currently available.

The example below stores a string into a custom created table within the TTF file.

```python
from stegosphere.containers.ttf import TTFContainer
from stegosphere.methods import ttf_CustomTable


font = TTFContainer('font.ttf')

embedded = ttf_CustomTable.embed(font, 'Encoded message!', table_name='STEG')
embedded.save('steg_font.ttf')

steg_font = TTFContainer('steg_font.ttf')

print(ttf_CustomTable.extract(steg_font, 'STEG'))

```

## wavelet transform
Integer Wavelet Transform is also available.
As it is not a steganography on its own, it needs to be combined with other techniques. In this example, IWT is applied to a RGB image and the high frequency coefficients in the diagonal direction
are used for LSB.
```python
from stegosphere.containers.image import ImageContainer
from stegosphere.methods import LSB, IWT
from stegosphere.io import *

img = ImageContainer('image.png')
px = img.read()
iwt_px, meta = IWT.transform(px, skip_last_axis=True)
hf = iwt_px[('1','1')]

cover = LSB.embed(hf, 'Embedded message!', method='delimiter', seed=42)

iwt_px[('1','1')] = cover
#transform pixels back into spatial domain
px_embed = IWT.inverse(iwt_px, meta)
img.flush(px_embed)
img.save('stego.png')
#extracing
steg_img = ImageContainer('stego.png')
steg_px = steg_img.read()
iwt, _ = IWT.transform(steg_px, skip_last_axis=True)
data = iwt[('1','1')]
uncover = LSB.extract(data, method='delimiter', seed=42)

print(binary_to_data(uncover))
#Expected output: 'Embedded message!'
```

## Multifile steganography
It is also possible to divide the payload across different files.
Different methods and parameters can be used for each file where data is being encoded.
Below, the content of the file `payload.png` is distributed randomly (with a seed) across the files `image.png` and `audio.wav`, with LSB and FVD being used.

```python
from stegosphere.containers import image, audio
from stegosphere.methods import LSB, VD
from stegosphere.tools import multifile
from stegosphere import io

data = io.file_to_binary('payload.png')

img = image.ImageContainer('image.png')
aud = audio.WAVContainer('audio.wav')

px = img.read()
frames = aud.read()

img_encode = lambda payload: LSB.embed(px, payload)
aud_encode = lambda payload: VD.embed(frames, payload)

#the encoded image pixels and audio frames
IMG, AUD = multifile.split_encode(data, [img_encode, aud_encode], seed=42)
#flush encoded data into file objects
img.flush(IMG)
aud.flush(AUD)

img.save('stego_image.png')
aud.save('stego_audio.wav')

#Extracting
img = image.ImageContainer('stego_image.png')
aud = audio.WAVContainer('stego_audio.wav')

px, frames = img.read(), aud.read()

img_extract = lambda: LSB.extract(px)
aud_extract = lambda: VD.extract(frames)

output = multifile.split_decode([img_extract, aud_extract], seed=42)

assert output == data

```
The payload can be distributed evenly (default setting),
using weighted distribution or roundrobin.

## File handling
Several functions for file handling are provided.
```
stegosphere.io.file_to_binary(path) --> converts any file into binary for encoding.

stegosphere.io.binary_to_file(binary_data, output_path) --> saves binary back into file format.

stegosphere.io.data_to_binary(data) --> converts any string into binary for encoding.

stegosphere.io.binary_to_data(binary) --> converts a binary string into a readable bytes object.
```

## Compression and encryption
Additionally, compression and encryption are provided.
Compression can be used by setting `compress='lzma'` when encoding/decoding. The given message will then be (de)compressed using lzma.

Compression can also be used on its own, by using `compression.compress`/`compression.decompress`. `lzma`and `deflate` algorithm are currently available.

## Research toolbox
The steganography and steganalysis modules can be combined to create research pipelines.
Below is an example of measuring how different metrics (speed, PSNR, ...) change when increasing the payload of an image, using LSB, for a set of images from a folder.
```python
import pandas as pd

from stegosphere.methods import LSB
from stegosphere.utils import generate_binary_payload as gbp
from stegosphere.analysis import efficiency, imperceptibility, detectability, accuracy
from stegosphere.containers import ContainerCollection
from stegosphere.containers import image

import os
path = os.listdir('PATH_TO_IMAGE_DIRECTORY')

images = ContainerCollection(path, image.ImageContainer, filter=True)
pxs = images.read()

df = pd.DataFrame(columns=['image', 'capacity','psnr','embed efficiency', 'detector','accuracy','extract efficiency'])

#test for different images
for path, px in zip(images.paths, pxs):
    max_capacity = LSB.max_capacity(px) - LSB.METADATA_LENGTH_LSB
    max_payload = gbp(max_capacity)

    #test for differrent payload sizes
    for cap in range(0, max_capacity, max_capacity//10):
        #Embedding efficiency
        with efficiency.Timer() as embed_time:
            emb = LSB.embed(px, max_payload[:cap])
        #Imperceptibility
        p = imperceptibility.psnr(px, emb, 255)
        #Detectability
        d = detectability.uniformity_detector(emb)
        #Extraction efficiency
        with efficiency.Timer() as extract_time:
            ext = LSB.extract(emb)
        #Extraction accuracy
        acc = accuracy.bit_error_rate(ext, max_payload[:cap])
        #Store results
        df.loc[len(df)] = path, cap, p , embed_time.diff, d, acc, extract_time.diff

print(df)

```

## Contributing
Any support or input is always welcomed.
Additional general methods are much needed.

Contact:

Email: maximilian.koch@student.uva.nl

LinkedIn: https://www.linkedin.com/in/maximilian-jw-koch/
