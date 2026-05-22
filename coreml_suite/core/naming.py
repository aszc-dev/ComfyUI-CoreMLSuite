"""Pure out_name composition for the Core ML UNet artifact.

Extracted from CoreMLConverter.convert in Phase 3 so the filename contract
can be tested + reused without instantiating the node. The string is the
cache key: every workflow that references a converted .mlpackage depends
on it staying byte-for-byte identical.
"""
from typing import Iterable, Tuple

ATTN_SUFFIX = {
    "SPLIT_EINSUM": "se",
    "SPLIT_EINSUM_V2": "se2",
    "ORIGINAL": "orig",
}


def compose_out_name(
    *,
    ckpt_name: str,
    batch_size: int,
    width: int,
    height: int,
    controlnet_support: bool,
    attention_implementation: str,
    lora_names: Iterable[str] = (),
) -> str:
    """Build the .mlpackage stem from convert() parameters.

    Locked behaviour (Phase 2 characterization tests):
      - first '.' in ckpt_name wins (`a.b.c.safetensors` -> `a`)
      - spaces collapse to underscores
      - LoRA names are taken stem-only, sorted, joined with '_' and
        prefixed with '_' when present (caller is expected to pass a
        sorted list; we sort defensively)
      - controlnet adds `_cn`
      - attn suffix is `_se` | `_se2` | `_orig`
    """
    stem = ckpt_name.split(".")[0]
    sorted_names = sorted(lora_names)
    lora_str = "_" + "_".join(name.split(".")[0] for name in sorted_names) if sorted_names else ""
    cn_suffix = "_cn" if controlnet_support else ""
    attn_suffix = "_" + ATTN_SUFFIX[attention_implementation]
    out_name = f"{stem}{lora_str}_{batch_size}x{width}x{height}{cn_suffix}{attn_suffix}"
    return out_name.replace(" ", "_")


def lora_names_from_params(lora_params: Iterable[Tuple[str, float]]) -> list[str]:
    """Mirror the sort applied inside CoreMLConverter.convert."""
    return [name for name, _ in sorted(lora_params, key=lambda pair: pair[0])]
