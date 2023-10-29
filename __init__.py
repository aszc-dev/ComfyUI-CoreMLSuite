import os
import sys

sys.path.append(os.path.dirname(__file__))

from coreml_suite.nodes import CoreMLLoaderUNet, CoreMLSampler, CoreMLModelAdapter
from coreml_suite.lcm import CoreMLConverterLCM, CoreMLSamplerLCM

NODE_CLASS_MAPPINGS = {
    "CoreMLUNetLoader": CoreMLLoaderUNet,
    "CoreMLSampler": CoreMLSampler,
    "CoreMLModelAdapter": CoreMLModelAdapter,
    "CoreMLSamplerLCM": CoreMLSamplerLCM,
    "CoreMLConverterLCM": CoreMLConverterLCM,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLUNetLoader": "Load Core ML UNet",
    "CoreMLSampler": "Core ML Sampler",
    "CoreMLModelAdapter": "Core ML Adapter (Experimental)",
    "CoreMLSamplerLCM": "Core ML LCM Sampler",
    "CoreMLConverterLCM": "Convert LCM to Core ML",
}
