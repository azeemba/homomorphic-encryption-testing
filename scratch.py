from functools import partial
import math
from multiprocessing import Pool
from time import perf_counter, process_time
from typing import Tuple

import numpy as np
from PIL import Image


def sobel_edge_detect(im: list[int], size: Tuple[int, int]) -> np.ndarray:
    rotated_mat_x = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    rorated_mat_y = [-1, -2, -1, 0, 0, 0, 1, 2, 1]

    W = size[0]
    H = size[1]

    out = np.zeros(W * H, dtype=np.int8)
    for i in range(W * H):
        y = i % W
        x = i // W
        if x in [0, W - 1] or y in [0, H - 1]:
            continue

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
        out[i] = min(int(math.sqrt(valx ** 2 + valy ** 2)), 255)
    return out

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
 
def sobel_edge_detect_parallel(im: list[int], size: Tuple[int, int]) -> np.ndarray:
    W = size[0]
    H = size[1]

    pool = Pool()
    res = pool.map(partial(handle_pixel, im, W), range(W*H))
    out = np.array(res, dtype=np.int8)
    return out



if __name__ == "__main__":
    im = Image.open("astro_512.png").convert("L")  # L is black and white

    start_perf = perf_counter()
    start_cpu = process_time()

    for _ in range(10):
        out_list = sobel_edge_detect_parallel(list(im.getdata()), im.size)

    end_perf = perf_counter()
    end_cpu = process_time()
    print(f"CPU: {end_cpu - start_cpu}. Wall: {end_perf - start_perf}")
    out = Image.fromarray(out_list.reshape(im.size), mode="L")
    out.save("result.png")
