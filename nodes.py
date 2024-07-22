import torch

from .matchering.log.handlers import set_handlers as log
from .matchering.core import process


class Matchering:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("AUDIO",),
                "reference": ("AUDIO",),
            }
        }

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return hash(frozenset(kwargs))

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = (
        "Result",
        "Result (no limiter)",
        "Result (no limiter, normalized)",
    )

    CATEGORY = "audio"
    FUNCTION = "matchering"

    def matchering(
        self,
        target,
        reference,
    ):
        log(print)

        result, result_no_limiter, result_no_limiter_normalized = process(
            target_audio=target,
            reference_audio=reference,
        )

        return (
            {
                "waveform": torch.from_numpy(result.T).unsqueeze(0),
                "sample_rate": reference["sample_rate"],
            },
            {
                "waveform": torch.from_numpy(result_no_limiter.T).unsqueeze(0),
                "sample_rate": reference["sample_rate"],
            },
            {
                "waveform": torch.from_numpy(result_no_limiter_normalized.T).unsqueeze(0),
                "sample_rate": reference["sample_rate"],
            },
        )


NODE_CLASS_MAPPINGS = {
    "Matchering": Matchering,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Matchering": "Matchering",
}
