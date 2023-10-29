import os
import sys

sys.path.append(os.path.dirname(__file__))

from coreml_suite import CoreMLLoaderUNet

NODE_CLASS_MAPPINGS = {
    "CoreMLUNetLoader": CoreMLLoaderUNet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLUNetLoader": "Load Core ML UNet"
}
