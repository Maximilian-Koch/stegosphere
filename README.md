# Stegosphere
A flexible, highly modular steganography library for image, audio, ttf, multiple file and all NumPy-array-readable steganography, including encryption and compression.

An initial documentation is available here: https://maximilian-koch.github.io/stegosphere/stegosphere.html#submodules

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
10. [Contributing](#contributing)

## General
The library was developed to allow for generalisation and compatability of different steganographical methods across file types.
The base steganography classes define steganography on top of numpy arrays, while the implementations for different file types primarily aid in converting between the file type and numpy arrays.

Currently, methods for image, audio, ttf and cross-file steganography are implemented.

## Image steganography
For image steganography, LSB (Least Significant Bit) and PVD (Pixel Value Differencing) steganography are currently available.

The example below loads an image, randomly distributes the message across the image using a seed and saves it.
```python
from stegosphere import image

img = image.LSB('image.png')
img.encode('Encoded message!', seed=42, method='delimiter')
img.save('stego_image.png')

steg_img = image.LSB('stego_image.png')
decoded_bits = steg_img.decode(seed=42, method='delimiter', compress=False)
print(image.binary_to_data(decoded_bits))
#Expected output: 'Encoded message!'
```
For additional parameters, see the chapter on [parameters](#additional-parameters).
### Audio steganography
For audio steganography, LSB (Least Significant Bit) and FVD (Frequency Value Differencing) steganography are currently available.
The example below loads an audio and encodes the file `image.png` into the audio. The image is then recovered and saved.
```python
from stegosphere import audio

wav = audio.FVD('audio.wav')
bin_image = audio.file_to_binary('image.png')
wav.encode(bin_image)
wav.save('steg_audio.wav')

steg_wav = audio.LSB('steg_audio.wav')
audio.binary_to_file(steg_wav.decode(), 'recovered_image.png')
```
For additional parameters, see the chapter on [parameters](#additional-parameters).

## ttf steganography
For ttf steganography, LSB (Least Significant Bit) and Custom Table steganography are currently available.

The example below stores a string into a custom created table within the TTF file.

```python
from stegosphere import ttf

font = ttf.CustomTable('the_font.ttf')
font.encode('Encoded message!', table_name='STEG')
font.save('stegano_font.ttf')

recover_font = ttf.CustomTable('stegano_font.ttf')
print(recover_font.decode(table_name='STEG'))
```

## Multimedia steganography
It is also possible to divide the payload across different files. For now only even distribution is possible.
Different methods and parameters can be used for each file where data is being encoded.

```python
from stegosphere import multimedia

data = file_to_binary('encode.png')

lsb_img = image.LSB('cover_image.png')
fvd_audio = audio.FVD('cover_audio.wav')

#Define the custom encoding functions
image_encode = lambda message: lsb_img.encode(message, seed=42, method='delimiter')
audio_encode = lambda message: fvd_audio.encode(message, seed=21)

#Encode the data evenly across the image and audio,
#with the data being randomly distributed using a seed.
split_encode(data, [image_encode,audio_encode], seed=100)


lsb_img.save('multimedia_stego.png')
fvd_audio.save('mutlimedia_stego.wav')

decode_lsb_img = image.LSB('multimedia_stego.png')
decode_fvd_audio = audio.FVD('multimedia_stego.wav')

image_decode = lambda: decode_lsb_img.decode(seed=42, method='delimiter')
audio_decode = lambda: decode_lsb_audio.decode(seed=21)

output = split_decode([image_decode, audio_decode], seed=100)

print(output==data)
```

## File handling
Several functions for file handling are provided.
```
stegosphere.file_to_binary(path) --> converts any file into binary for encoding.

stegosphere.binary_to_file(binary_data, output_path) --> saves binary back into file format.

stegosphere.data_to_binary(data) --> converts any string into binary for encoding.

stegosphere.binary_to_data(binary) --> converts a binary string into a readable bytes object.
```

## Compression and encryption
Additionally, compression and encryption are provided.
Compression can be used by setting `compress=True` when encoding/decoding. The given message will then be (de)compressed using lzma.


## Additional parameters
| Parameter     | Available for     | Effect     |
|--------------|--------------|--------------|
| `seed`  | LSB, VD, multimedia  | Distributes payload pseudorandomly across the file. Reduces detectability drastically.  |
| `matching`  | in development for LSB  | less detectable way of adapting bits in LSB  |
| `bits` | LSB  | increases capacity, increases detectability  |
| `method`  | LSB, VD  | The method to detect end of message when decoding. Either 'delimiter', 'metadata' or None.  |
| `metadata_length`  | LSB, VD  | Bits used at the beginning of the message for the metadata. Change only needed for very short or very long (>0.5GB) payloads.  |
| `delimiter_message`  | LSB, VD  | The message used as an end of message signifier when decoding.  |
| `compress`  | LSB, VD  | Compress message to save space when encoding.  |

## More file types
Any file types which can be read or converted as a numpy array can be used for some of the steganographic methods, which are implemented in `stegosphere.spatial`.

## Contributing
Any support or input is always welcomed.
Additional file type support and general methods are needed.

Contact:
email: maximilian.koch@student.uva.nl

LinkedIn: https://www.linkedin.com/in/maximilian-jw-koch/
