import numpy as np
import torch

from comfy.sd import CLIP
from comfy.sd1_clip import SDClipModel


class CoreMLCLIP(CLIP):
    pass


class SDClipModelCoreML(SDClipModel):
    def __init__(self, **kwargs):
        self.model = kwargs.pop("coreml_model")
        super().__init__(**kwargs)

    def encode_token_weights(self, token_weight_pairs):
        tokens = np.array(token_weight_pairs["l"]).astype(np.float16)[:, :, 0]
        encoded = self.model(input_ids=tokens)
        cond = torch.from_numpy(encoded["last_hidden_state"])
        pooled = torch.from_numpy(encoded["pooled_outputs"])
        return cond, pooled
