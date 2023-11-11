import os
import sys

sys.path.append(os.path.dirname(__file__))

from coreml_suite.nodes import CoreMLLoaderUNet, CoreMLSampler, CoreMLModelAdapter
from coreml_suite.lcm import (
    COREML_CONVERT_LCM,
)

NODE_CLASS_MAPPINGS = {
    "CoreMLUNetLoader": CoreMLLoaderUNet,
    "CoreMLSampler": CoreMLSampler,
    "CoreMLModelAdapter": CoreMLModelAdapter,
    "Core ML LCM Converter": COREML_CONVERT_LCM,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLUNetLoader": "Load Core ML UNet",
    "CoreMLSampler": "Core ML Sampler",
    "CoreMLModelAdapter": "Core ML Adapter (Experimental)",
    "Core ML LCM Converter": "Convert LCM to Core ML",
}
