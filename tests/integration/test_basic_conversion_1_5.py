import json
import os

import pytest
import requests
import torch
from PIL import Image
import numpy as np

from folder_paths import get_save_image_path, get_output_directory

IMAGE_PREFIX = "E2E-1.5"
IMAGE_PREFIX_CML = f"{IMAGE_PREFIX}-CoreML"
IMAGE_PREFIX_MPS = f"{IMAGE_PREFIX}-MPS"


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

    def get_latest_image(self, prefix):
        full_output_folder, images = self.list_images()
        for image in sorted(images, reverse=True):
            if image.startswith(prefix):
                return os.path.join(full_output_folder, image)
        return None


@pytest.fixture(scope="function")
def output_image_repository():
    repo = OutputImageRepository(IMAGE_PREFIX)
    yield repo
    repo.delete_images()


def test_basic_conversion_1_5(output_image_repository):
    with open("tests/integration/workflows/e2e-1.5-basic-conversion.json") as f:
        prompt = json.load(f)
    prompt = randomize_seed_in_prompt(prompt)
    queue_prompt(prompt)

    coreml_img_path = output_image_repository.get_latest_image(IMAGE_PREFIX_CML)
    mps_img_path = output_image_repository.get_latest_image(IMAGE_PREFIX_MPS)

    coreml_image = Image.open(coreml_img_path)
    mps_image = Image.open(mps_img_path)

    assert psnr(np.array(coreml_image), np.array(mps_image)) > 25


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


def randomize_seed_in_prompt(prompt):
    seed = torch.random.seed()
    prompt["3"]["inputs"]["seed"] = seed
    prompt["11"]["inputs"]["seed"] = seed
    return prompt
