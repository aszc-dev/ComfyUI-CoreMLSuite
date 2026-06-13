"""LCM runtime support (sampler-side).

The dedicated LCM converter node was removed once the standard ``CoreMLConverter``
gained model-version auto-detection (full-distill LCM is detected from the
checkpoint). What remains here is runtime sampling support — ``utils`` patches the
model sampling and supplies the guidance embedding when a converted UNet exposes
``timestep_cond``.
"""
