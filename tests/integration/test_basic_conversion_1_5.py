import json
import os

import pytest
import requests
from PIL import Image
import numpy as np

from folder_paths import get_save_image_path, get_output_directory

IMAGE_PREFIX = "E2E-1.5-CoreML"


class OutputImageRepository:
    def __init__(self, name_prefix):
        self.name_prefix = name_prefix

    def list_images(self):
        full_output_folder, _, _, _, _ = get_save_image_path(
            self.name_prefix, get_output_directory(), 512, 512
        )
        return full_output_folder, os.listdir(full_output_folder)

    def delete_images(self):
        full_output_folder, images = self.list_images()
        for image in images:
            os.remove(os.path.join(full_output_folder, image))


@pytest.fixture(scope="module")
def output_image_repository():
    repo = OutputImageRepository(IMAGE_PREFIX)
    yield repo
    repo.delete_images()


def test_basic_conversion_1_5(output_image_repository):
    with open("integration/workflows/e2e-1.5-basic-conversion.json") as f:
        prompt = json.load(f)
    queue_prompt(prompt)

    full_output_folder, images = output_image_repository.list_images()
    assert len(images) == 2
    assert all(image.startswith(IMAGE_PREFIX) for image in images)
    assert all(image.endswith(".png") for image in images)
    assert all(
        os.path.isfile(os.path.join(full_output_folder, image)) for image in images
    )

    image1 = Image.open(os.path.join(full_output_folder, images[0]))
    image2 = Image.open(os.path.join(full_output_folder, images[1]))
    assert psnr(np.array(image1), np.array(image2)) > 30
    assert psnr(np.array(image2), np.array(image1)) > 30


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def queue_prompt(prompt: dict):
    p = {"prompt": prompt}
    data = json.dumps(p).encode("utf-8")
    req = requests.post("http://localhost:8188/prompt", data=data)
    assert req.status_code == 200
    while True:
        req = requests.get("http://localhost:8188/prompt")
        if req.json()["exec_info"]["queue_remaining"] == 0:
            break
