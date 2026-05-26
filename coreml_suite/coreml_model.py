import time

import coremltools as ct

from coreml_suite.logger import logger


class CoreMLModel:
    """Small runtime wrapper around coremltools.models.MLModel.

    This keeps the inference path independent from apple/ml-stable-diffusion's
    CoreMLModel wrapper while preserving the contract used by the sampler code:
    ``expected_inputs`` and callable prediction.
    """

    def __init__(self, model_path, compute_unit):
        self.model_path = model_path
        self.compute_unit = self._compute_unit(compute_unit)

        logger.info(f"Loading {model_path} to {self.compute_unit.name}")
        start = time.time()
        self.model = ct.models.MLModel(model_path, compute_units=self.compute_unit)
        logger.info(f"Loading {model_path} took {time.time() - start:.1f} seconds")

        self.expected_inputs = self._expected_inputs()

    def __call__(self, **kwargs):
        return self.model.predict(kwargs)

    @staticmethod
    def _compute_unit(compute_unit):
        if isinstance(compute_unit, ct.ComputeUnit):
            return compute_unit
        return ct.ComputeUnit[compute_unit]

    def _expected_inputs(self):
        return {
            feature.name: {
                "shape": tuple(feature.type.multiArrayType.shape),
            }
            for feature in self.model.get_spec().description.input
        }
