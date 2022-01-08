from typing import cast

import numpy as np
import requests
from PIL import Image


def load_image() -> Image:
    im = Image.open("astro_512.png").convert("L")  # L is black and white
    return im


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


def save_result_image(size, response: requests.Response):
    body = response.json()
    pixels = body["pixels"]
    print(pixels[200*512 + 200])
    arr = np.array(pixels, dtype=np.uint8)
    out = Image.fromarray(arr.reshape(size))
    out.save("result.png")


def main():
    im = load_image()
    response = send_edge_detect_request(im)
    save_result_image(im.size, response)


if __name__ == "__main__":
    main()
