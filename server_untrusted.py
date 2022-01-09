import base64
import math
from functools import partial
from multiprocessing import Pool
from time import perf_counter
from typing import Tuple

from flask import Flask, request
from Pyfhel import PyCtxt, Pyfhel

app = Flask(__name__)


@app.route("/")
def home():
    return "Works"


def handle_pixel(im: list[int], W: int, i: int) -> int:
    """Helper for parallelization"""
    rotated_mat_x = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    rorated_mat_y = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    y = i % W
    x = i // W
    if x in [0, W - 1] or y in [0, W - 1]:
        return 0

    # Order: -W-1, -W, -W+1  -1 0 +1  W-1 W W+1
    #         0 1 2 3 4 5 6 7 8
    # (N % 3) -1
    # (N // 3)
    valx = 0
    valy = 0
    for j in range(9):
        index = (j % 3) - 1 + ((j // 3) - 1) * W + i
        valx += im[index] * rotated_mat_x[j]
        valy += im[index] * rorated_mat_y[j]
    return min(int(math.sqrt(valx ** 2 + valy ** 2)), 255)


def sobel_edge_detect_parallel(im: list[int], size: Tuple[int, int]) -> list[int]:
    W = size[0]
    H = size[1]

    pool = Pool()
    res = pool.map(partial(handle_pixel, im, W), range(W * H))
    return res


def sobel_encrypted_edge_detect(
    enc_context: str, enc_public: str, pixels: list[str], size: int
) -> list[str]:
    enc = Pyfhel()
    enc.from_bytes_context(base64.b64decode(enc_context))
    enc.from_bytes_publicKey(base64.b64decode(enc_public))

    W = size
    H = size

    results = []
    for i in range(len(pixels)):
        cur = pixels[i]
        y = i % W
        x = i // W
        cur_bytes = base64.b64decode(cur)
        cipherobj = PyCtxt(pyfhel=enc, serialized=cur_bytes, encoding="float")
        results.append(cipherobj)

    def cipher_to_str(cipher):
        return base64.b64encode(cipher.to_bytes()).decode("ascii")

    zero = enc.encryptFrac(0)
    zero_out = cipher_to_str(zero)
    edged = []
    for i in range(len(results)):
        y = i % W
        x = i // W
        if x in [0, W - 1] or y in [0, W - 1]:
            edged.append([zero_out, zero_out])
            continue

        # Order:
        # -W-1, -W, -W+1
        # -1 0 +1
        #  W-1 W W+1
        valx = (
            results[i - W + 1]
            - results[i - W - 1]
            - results[i - 1]
            - results[i - 1]
            + results[i + 1]
            + results[i + 1]
            - results[i + W - 1]
            + results[i + W + 1]
        )
        valy = (
            results[i + W - 1]
            + results[i + W]
            + results[i + W]
            + results[i + W + 1]
            - results[i - W - 1]
            - results[i - W]
            - results[i - W]
            - results[i - W + 1]
        )

        edged.append([cipher_to_str(valx), cipher_to_str(valy)])

    return edged


@app.route("/detect_edge", methods=["POST"])
def handle_detect_edge():
    """Expected body:
    - encrypted: true/false
    - size: int (assume square image)
    - pixels: list[int] (can be encrypted)
    - public_key: ?
    """
    body = request.json
    pixels = body["pixels"]
    size = body["size"]

    start_perf = perf_counter()
    if body["encrypted"]:
        im = sobel_encrypted_edge_detect(
            body["context"], body["public_key"], pixels, size
        )
    else:
        im = sobel_edge_detect_parallel(pixels, (size, size))
    print(f"Processing: {perf_counter() - start_perf} seconds")
    return {"pixels": im}


app.run()
