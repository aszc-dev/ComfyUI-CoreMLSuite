import os

import folder_paths
from comfy import sd, utils


def load_lora(lora_params, ckpt_name):
    lora_params = (
        lora_params.copy()
        if isinstance(lora_params, (list, dict, set))
        else lora_params
    )
    ckpt_name = (
        ckpt_name.copy() if isinstance(ckpt_name, (list, dict, set)) else ckpt_name
    )

    def recursive_load_lora(lora_params, clip):
        if len(lora_params) == 0:
            return clip

        lora_name, strength_model, strength_clip = lora_params[0]
        if os.path.isabs(lora_name):
            lora_path = lora_name
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)

        _, lora_clip = sd.load_lora_for_models(
            None, clip, utils.load_torch_file(lora_path), strength_model, strength_clip
        )

        # Call the function again with the new lora_model and lora_clip and the remaining tuples
        return recursive_load_lora(lora_params[1:], lora_clip)

    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    _, clip, _, _ = sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=False,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )

    lora_clip = recursive_load_lora(lora_params, clip)

    return lora_clip
