"""Pure SDXL detection + time_ids/text_embeds assembly.

The framework-coupled adapter `add_sdxl_model_options` lives in models.py
and delegates the math here. Characterization tests cover base (len 6) vs
refiner (len 5) and the closure free-vars produced by
`sdxl_model_function_wrapper`.
"""
import torch


def is_sdxl(coreml_model):
    return (
        "time_ids" in coreml_model.expected_inputs
        and "text_embeds" in coreml_model.expected_inputs
    )


def is_sdxl_base(coreml_model):
    return (
        is_sdxl(coreml_model)
        and coreml_model.expected_inputs["time_ids"]["shape"][1] == 6
    )


def is_sdxl_refiner(coreml_model):
    return (
        is_sdxl(coreml_model)
        and coreml_model.expected_inputs["time_ids"]["shape"][1] == 5
    )


def build_sdxl_time_ids(pos_dict, neg_dict, *, is_base: bool, is_refiner: bool):
    """Compose the (2, N) time_ids tensor for the SDXL Core ML UNet.

    - base: N=6  -> [h, w, crop_h, crop_w, target_h, target_w]
    - refiner: N=5 -> [h, w, crop_h, crop_w, aesthetic_score]
    - neither: N=4 -> [h, w, crop_h, crop_w] (edge case kept for parity)
    """
    pos_time_ids = [
        pos_dict.get("height", 768),
        pos_dict.get("width", 768),
        pos_dict.get("crop_h", 0),
        pos_dict.get("crop_w", 0),
    ]
    neg_time_ids = [
        neg_dict.get("height", 768),
        neg_dict.get("width", 768),
        neg_dict.get("crop_h", 0),
        neg_dict.get("crop_w", 0),
    ]

    if is_base:
        pos_time_ids += [
            pos_dict.get("target_height", 768),
            pos_dict.get("target_width", 768),
        ]
        neg_time_ids += [
            neg_dict.get("target_height", 768),
            neg_dict.get("target_width", 768),
        ]

    if is_refiner:
        pos_time_ids += [pos_dict.get("aesthetic_score", 6)]
        neg_time_ids += [neg_dict.get("aesthetic_score", 2.5)]

    return torch.tensor([pos_time_ids, neg_time_ids])


def build_sdxl_text_embeds(pos_pooled, neg_pooled):
    """Concat pos then neg along the batch dim. Locked contract."""
    return torch.cat((pos_pooled, neg_pooled))


def sdxl_model_function_wrapper(time_ids, text_embeds, refiner=False):
    def wrapper(model_function, params):
        x = params["input"]
        t = params["timestep"]
        c = params["c"]

        context = c.get("c_crossattn")

        if context is None:
            return torch.zeros_like(x)

        if refiner and context is not None:
            # converted refiner accepts only g clip
            c["c_crossattn"] = context[:, :, 768:]

        return model_function(x, t, **c, time_ids=time_ids, text_embeds=text_embeds)

    return wrapper
