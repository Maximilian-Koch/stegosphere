#Demo!
if __name__ == '__main__':
    from stegosphere import image, embeddings, binary_to_data
    img = image.ImageContainer('logo_small_stego.png')
    px = img.read()
    payload = embeddings.LSB.extract(px)
    readable_payload = binary_to_data(payload).decode()
    if 'stegosphere' in readable_payload:
        print(readable_payload)
    else:
        print('Decoding the message did not succeed.')