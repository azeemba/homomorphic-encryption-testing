import math
from functools import partial
from multiprocessing import Pool
from time import perf_counter, process_time
from typing import Tuple

import numpy as np
from flask import Flask, request

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
    start_perf = perf_counter()
    res = pool.map(partial(handle_pixel, im, W), range(W * H))
    print(f"Processing: {perf_counter() - start_perf} seconds")
    return res


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
    if body["encrypted"]:
        return "Not implemented yet", 500
    else:
        im = sobel_edge_detect_parallel(pixels, (size, size))
        return {"pixels": im}


app.run()
