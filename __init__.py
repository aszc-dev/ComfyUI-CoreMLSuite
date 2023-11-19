import os
import sys

sys.path.append(os.path.dirname(__file__))

from coreml_suite.nodes import (
    CoreMLLoaderUNet,
    CoreMLSampler,
    CoreMLSamplerAdvanced,
    CoreMLModelAdapter,
    COREML_CONVERT,
    COREML_LOAD_LORA,
)
from coreml_suite.lcm import (
    COREML_CONVERT_LCM,
)

NODE_CLASS_MAPPINGS = {
    "CoreMLUNetLoader": CoreMLLoaderUNet,
    "CoreMLSampler": CoreMLSampler,
    "CoreMLSamplerAdvanced": CoreMLSamplerAdvanced,
    "CoreMLModelAdapter": CoreMLModelAdapter,
    "Core ML LoRA Loader": COREML_LOAD_LORA,
    "Core ML Converter": COREML_CONVERT,
    "Core ML LCM Converter": COREML_CONVERT_LCM,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLUNetLoader": "Load Core ML UNet",
    "CoreMLSampler": "Core ML Sampler",
    "CoreMLSamplerAdvanced": "Core ML Sampler (Advanced)",
    "CoreMLModelAdapter": "Core ML Adapter (Experimental)",
    "Core ML LoRA Loader": "Load LoRA to use with Core ML",
    "Core ML Converter": "Convert Checkpoint to Core ML",
    "Core ML LCM Converter": "Convert LCM to Core ML",
}
