import os.path

from coremltools import ComputeUnit
from python_coreml_stable_diffusion.coreml_model import CoreMLModel

import folder_paths

from coreml_suite.logger import logger


class CoreMLLoader:
    PACKAGE_DIRNAME = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coreml_name": (list(s.coreml_filenames().keys()),),
                "compute_unit": (
                    [
                        ComputeUnit.CPU_AND_NE.name,
                        ComputeUnit.CPU_AND_GPU.name,
                        ComputeUnit.ALL.name,
                        ComputeUnit.CPU_ONLY.name,
                    ],
                ),
            }
        }

    FUNCTION = "load"
    CATEGORY = "Core ML Suite"

    @classmethod
    def coreml_filenames(cls):
        extensions = (".mlmodelc", ".mlpackage")
        all_paths = folder_paths.get_filename_list_(cls.PACKAGE_DIRNAME)[1]
        coreml_paths = folder_paths.filter_files_extensions(all_paths, extensions)

        return {os.path.split(p)[-1]: p for p in coreml_paths}

    def load(self, coreml_name, compute_unit):
        logger.info(f"Loading {coreml_name} to {compute_unit}")

        coreml_path = self.coreml_filenames()[coreml_name]

        sources = "compiled" if coreml_name.endswith(".mlmodelc") else "packages"

        return self._load(coreml_path, compute_unit, sources)

    def _load(self, coreml_path, compute_unit, sources):
        return (CoreMLModel(coreml_path, compute_unit, sources),)


class CoreMLLoaderCkpt(CoreMLLoader):
    PACKAGE_DIRNAME = "checkpoints"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")

    def load(self, coreml_name, compute_unit):
        # TODO: Implement this
        pass


class CoreMLLoaderTextEncoder(CoreMLLoader):
    PACKAGE_DIRNAME = "clip"
    RETURN_TYPES = ("CLIP",)

    def load(self, coreml_name, compute_unit):
        # TODO: Implement this
        pass


class CoreMLLoaderUNet(CoreMLLoader):
    PACKAGE_DIRNAME = "unet"
    RETURN_TYPES = ("COREML_UNET",)
    RETURN_NAMES = ("coreml_model",)


class CoreMLLoaderVAE(CoreMLLoader):
    PACKAGE_DIRNAME = "vae"
    RETURN_TYPES = ("VAE",)

    def load(self, coreml_name, compute_unit):
        # TODO: Implement this
        pass
