import base64
import gc
import math
from functools import partial, wraps
from multiprocessing import Pool
from time import perf_counter

import numpy as np
import requests
from PIL import Image
from Pyfhel import PyCtxt, Pyfhel, PyPtxt


def time_me(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        start = perf_counter()
        result = f(*args, **kwargs)
        print(f"{f.__name__} took {perf_counter() - start} seconds")
        return result

    return decorated


def load_image() -> Image:
    fname = "astro_256.png"
    im = Image.open(fname).convert("L")  # L is black and white
    return im


def encrypt_integer_chunk(pub_key, pub_context, pixels, chunksize, start) -> list[str]:
    # print("In Encrypt_integer ", len(pub_key), len(pub_context))
    enc = Pyfhel()
    enc.from_bytes_context(pub_context)
    enc.from_bytes_publicKey(pub_key)
    # print("Constructed Pyfhel object")

    def help(i) -> str:
        px = pixels[i]
        encrypted_px = enc.encryptFrac(px).to_bytes()
        return base64.b64encode(encrypted_px).decode("ascii")

    result = list(map(help, range(start, start + chunksize)))
    # print("Out Encrypt_integer", len(pub_key), len(pub_context))
    return result


@time_me
def encrypt_image(enc: Pyfhel, im: Image) -> list[str]:
    imdata: list[float] = [x / 255.0 for x in im.getdata()]
    manual_chunk = 128
    length = len(imdata)
    with Pool() as p:
        pixel_chunks = p.map(
            partial(
                encrypt_integer_chunk,
                enc.to_bytes_publicKey(),
                enc.to_bytes_context(),
                imdata,
                manual_chunk,
            ),
            range(0, length, manual_chunk),
        )

    pixels: list[str] = []
    for chunk in pixel_chunks:
        pixels += chunk
    return pixels


def send_encrypted_request(enc: Pyfhel, im: Image):
    pub_bytes: bytes = enc.to_bytes_publicKey()
    pub_context: bytes = enc.to_bytes_context()
    print("Starting to encrypt file")
    req = {
        "encrypted": True,
        "size": im.size[0],
        "pixels": encrypt_image(enc, im),
        "public_key": base64.b64encode(pub_bytes).decode("ascii"),
        "context": base64.b64encode(pub_context).decode("ascii"),
    }

    def handle_encrypted_val(encrypted):
        x = base64.b64decode(encrypted)
        decoded_px = PyCtxt()
        decoded_px.from_bytes(x, "float")
        return enc.decryptFrac(decoded_px)

    out_pix = []
    with requests.post(
        "http://localhost:5000/detect_edge", json=req, stream=True
    ) as response:
        req = None
        gc.collect()
        for line in response.iter_lines(chunk_size=128000):
            elements = line.decode("utf-8").split(" ")
            decrypted_x = handle_encrypted_val(elements[0])
            decrypted_y = handle_encrypted_val(elements[1])
            result = math.sqrt(decrypted_y * decrypted_y + decrypted_x * decrypted_x)
            out_pix.append(min(int(result * 255), 255))

    return out_pix


def send_edge_detect_request(im: Image) -> requests.Response:
    req = {
        "encrypted": False,
        "size": im.size[0],
        "pixels": list(im.getdata()),
        "public_key": "notyet",
    }
    resp = requests.post("http://localhost:5000/detect_edge", json=req)
    resp.raise_for_status()
    return resp


def save_result_image(size, pixels: list[int], name="result.png"):
    arr = np.array(pixels, dtype=np.uint8)
    out = Image.fromarray(arr.reshape(size))
    out.save("result.png")


@time_me
def try_plaintext_edge_detection(im: Image):
    response = send_edge_detect_request(im)
    body = response.json()
    pixels = body["pixels"]
    save_result_image(im.size, pixels, "result-plain.png")


@time_me
def try_encrypted_edge_detection(im: Image):
    enc = Pyfhel()
    enc.contextGen(65537)
    enc.keyGen()
    out_pix = send_encrypted_request(enc, im)
    save_result_image(im.size, out_pix)


def main():
    im = load_image()
    try_encrypted_edge_detection(im)


if __name__ == "__main__":
    main()
