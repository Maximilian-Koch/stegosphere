# Stegosphere
A flexible, highly modular steganography and steganalysis library for image, audio, ttf, multiple file and all NumPy-array-readable steganography, including encryption and compression.

pip package is coming in February. For now, local installation or running from a folder outside is needed.

It is meant to be usable for research by combining steganography and steganalysis, see [Research toolbox](#research-toolbox).


# Table of contents
1. [General usage](#general)
2. [Image steganography](#image-steganography)
3. [Audio steganography](#audio-steganography)
4. [ttf steganography](#ttf-steganography)
5. [Multimedia steganography](#multimedia-steganography)
6. [File handling](#file-handling)
7. [Compression and Encryption](#compression-and-encryption)
8. [Additional parameters](#additional-parameters)
9. [More file types](#more-file-types)
10. [Research toolbox](#research-toolbox)
11. [Contributing](#contributing)

## General
The library is made to allow for generalisation and compatability of different steganographical methods across file types.
The base steganography classes define steganography on top of numpy arrays, while the implementations for different file types primarily aid in converting between the file type and numpy arrays.

Currently, methods for image, audio, ttf, video and multi-file steganography are implemented.

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
For additional parameters, see the chapter on [parameters](#additional-parameters).
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

## Additional parameters
| Parameter     | Available for     | Effect     |
|--------------|--------------|--------------|
| `seed`  | LSB, VD, multifile  | Distributes payload pseudorandomly across the file. Reduces detectability drastically.  |
| `matching`  | in development for LSB  | less detectable way of adapting bits in LSB  |
| `bits` | LSB  | increases capacity, increases detectability  |
| `method`  | LSB, VD  | The method to detect end of message when decoding. Either 'delimiter', 'metadata' or None.  |
| `metadata_length`  | LSB, VD  | Bits used at the beginning of the message for the metadata. Change only needed for very short or very long (>0.5GB) payloads.  |
| `delimiter_message`  | LSB, VD  | The message used as an end of message signifier when decoding.  |
| `compress`  | LSB, VD  | Compress message to save space when encoding.  |

## More file types
Any file types which can be read or converted as a numpy array can be used for some of the steganographic methods, which are implemented in the `methods` folder.

## Research toolbox
The steganography and steganalysis modules can be combined to create research pipelines.
Below is an example of measuring how different measures change when increasing the payload of an image, using LSB.
```python
import numpy as np
import pandas as pd
import time

from stegosphere.containers.image import ImageContainer
from stegosphere.methods import LSB
from stegosphere.utils import generate_binary_payload as gbp
from stegosphere.analysis.imperceptibility import mse, psnr
from stegosphere.analysis.detectability import random_detector, uniformity_detector
from stegosphere.analysis.accuracy import extract_accuracy

img = ImageContainer('image.png')
px = img.read()

#longest possible payload for that image
capacity = LSB.max_capacity(px)-LSB.METADATA_LENGTH_LSB
max_payload = gbp(capacity)

df = pd.DataFrame(columns=['capacity','psnr','embed efficiency', 'detector','accuracy','extract efficiency'])
LSB.BACKEND = True

#test for different lengths
for cap in range(1, 100000, 5000):
    t1 = time.time()
    emb = LSB.embed(px, max_payload[:cap])
    t2 = time.time()
    p = psnr(px, emb, 255)
    d = uniformity_detector(emb)
    
    t3 = time.time()
    ext = LSB.extract(emb)
    t4 = time.time()
    acc = extract_accuracy(ext,max_payload[:cap])
    df.loc[len(df)] = cap, p, t2-t1, d, acc, t4-t3

df
```

## Contributing
Any support or input is always welcomed.
Additional general methods are much needed.

Contact:
email: maximilian.koch@student.uva.nl

LinkedIn: https://www.linkedin.com/in/maximilian-jw-koch/
