from typing import Tuple
import math

import numpy as np
from PIL import Image


def sobel_edge_detect(im: list[int], size: Tuple[int, int]) -> np.ndarray:
    rotated_mat_x = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    rorated_mat_y = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    # rotated_mat_x = [0]*9
    # rotated_mat_x[4] = 1
    # rorated_mat_y = rotated_mat_x

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


if __name__ == "__main__":
    im = Image.open("astro_512.png").convert("L")  # L is black and white

    out_list = sobel_edge_detect(list(im.getdata()), im.size)
    out = Image.fromarray(out_list.reshape(im.size), mode="L")
    out.save("result.png")
