import os
import sys

sys.path.append(os.path.dirname(__file__))

from coreml_suite.nodes import (
    CoreMLLoaderUNet,
    CoreMLSampler,
    CoreMLModelAdapter,
    COREML_CONVERT,
    COREML_LOAD_CLIP,
)
from coreml_suite.lcm import (
    COREML_CONVERT_LCM,
)

NODE_CLASS_MAPPINGS = {
    "CoreMLUNetLoader": CoreMLLoaderUNet,
    "Core ML CLIP Loader": COREML_LOAD_CLIP,
    "CoreMLSampler": CoreMLSampler,
    "CoreMLModelAdapter": CoreMLModelAdapter,
    "Core ML Converter": COREML_CONVERT,
    "Core ML LCM Converter": COREML_CONVERT_LCM,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLUNetLoader": "Load Core ML UNet",
    "CoreMLSampler": "Core ML Sampler",
    "Core ML CLIP Loader": "Load Core ML CLIP",
    "CoreMLModelAdapter": "Core ML Adapter (Experimental)",
    "Core ML Converter": "Convert Checkpoint to Core ML",
    "Core ML LCM Converter": "Convert LCM to Core ML",
}
