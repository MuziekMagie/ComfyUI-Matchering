import torch

from .matchering.log.handlers import set_handlers as log
from .matchering.core import process
from .matchering.defaults import Config, LimiterConfig


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

    CATEGORY = "audio/matchering"
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
                "waveform": torch.from_numpy(result_no_limiter_normalized.T).unsqueeze(
                    0
                ),
                "sample_rate": reference["sample_rate"],
            },
        )


class MatcheringAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("AUDIO",),
                "reference": ("AUDIO",),
                "internal_sample_rate": (
                    "INT",
                    {"default": 44100, "min": 0, "max": 192000, "step": 1},
                ),
                "max_length": ("FLOAT", {"default": 15 * 60, "min": 0, "step": 1}),
                "max_piece_size": ("FLOAT", {"default": 15, "min": 0, "step": 1}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": (2**15 - 61) / 2**15,
                        "max": 0.9999999,
                        "step": 0.0000001,
                        "round": False,
                    },
                ),
                "min_value": (
                    "FLOAT",
                    {"default": 1e-6, "min": 0, "max": 0.1, "step": 1e-6},
                ),
                "fft_size": ("INT", {"default": 4096, "min": 1, "step": 1}),
                "lin_log_oversampling": ("INT", {"default": 4, "min": 1, "step": 1}),
                "rms_correction_steps": ("INT", {"default": 4, "min": 0, "step": 1}),
                "clipping_samples_threshold": (
                    "INT",
                    {"default": 8, "min": 0, "step": 1},
                ),
                "limited_samples_threshold": (
                    "INT",
                    {"default": 128, "min": 0, "step": 1},
                ),
                "allow_equality": ("BOOLEAN", {"default": False}),
                "lowess_frac": (
                    "FLOAT",
                    {"default": 0.0375, "min": 0.0001, "step": 0.0001},
                ),
                "lowess_it": ("INT", {"default": 0, "min": 0, "step": 1}),
                "lowess_delta": ("FLOAT", {"default": 0.001, "min": 0, "step": 0.001}),
            },
            "optional": {
                "limiter_config": ("MATCHERING_LIMITER_CONFIG",),
            },
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

    CATEGORY = "audio/matchering"
    FUNCTION = "matchering_advanced"

    def matchering_advanced(
        self,
        target,
        reference,
        internal_sample_rate,
        max_length,
        max_piece_size,
        threshold,
        min_value,
        fft_size,
        lin_log_oversampling,
        rms_correction_steps,
        clipping_samples_threshold,
        limited_samples_threshold,
        allow_equality,
        lowess_frac,
        lowess_it,
        lowess_delta,
        limiter_config=LimiterConfig(),
    ):
        log(print)

        result, result_no_limiter, result_no_limiter_normalized = process(
            target_audio=target,
            reference_audio=reference,
            config=Config(
                internal_sample_rate=internal_sample_rate,
                max_length=max_length,
                max_piece_size=max_piece_size,
                threshold=threshold,
                min_value=min_value,
                fft_size=fft_size,
                lin_log_oversampling=lin_log_oversampling,
                rms_correction_steps=rms_correction_steps,
                clipping_samples_threshold=clipping_samples_threshold,
                limited_samples_threshold=limited_samples_threshold,
                allow_equality=allow_equality,
                lowess_frac=lowess_frac,
                lowess_it=lowess_it,
                lowess_delta=lowess_delta,
                limiter=limiter_config,
            ),
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
                "waveform": torch.from_numpy(result_no_limiter_normalized.T).unsqueeze(
                    0
                ),
                "sample_rate": reference["sample_rate"],
            },
        )


class MatcheringLimiterConfig:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attack": ("FLOAT", {"default": 1, "min": 0.1, "step": 0.1}),
                "hold": ("FLOAT", {"default": 1, "min": 0.1, "step": 0.1}),
                "release": ("FLOAT", {"default": 3000, "min": 1, "step": 1}),
                "attack_filter_coefficient": (
                    "FLOAT",
                    {"default": -2, "min": -1000, "step": 0.1},
                ),
                "hold_filter_order": (
                    "INT",
                    {"default": 1, "min": 1, "step": 1},
                ),
                "hold_filter_coefficient": (
                    "FLOAT",
                    {"default": 7, "step": 0.1},
                ),
                "release_filter_order": (
                    "INT",
                    {"default": 1, "min": 1, "step": 1},
                ),
                "release_filter_coefficient": ("FLOAT", {"default": 800, "step": 1}),
            },
        }

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return hash(frozenset(kwargs))

    RETURN_TYPES = ("MATCHERING_LIMITER_CONFIG",)
    RETURN_NAMES = ("limiter_config",)

    CATEGORY = "audio/matchering"
    FUNCTION = "matchering_limiter_config"

    def matchering_limiter_config(
        self,
        attack,
        hold,
        release,
        attack_filter_coefficient,
        hold_filter_order,
        hold_filter_coefficient,
        release_filter_order,
        release_filter_coefficient,
    ):

        limiter_config = LimiterConfig(
            attack=attack,
            hold=hold,
            release=release,
            attack_filter_coefficient=attack_filter_coefficient,
            hold_filter_order=hold_filter_order,
            hold_filter_coefficient=hold_filter_coefficient,
            release_filter_order=release_filter_order,
            release_filter_coefficient=release_filter_coefficient,
        )

        return (limiter_config,)


NODE_CLASS_MAPPINGS = {
    "Matchering": Matchering,
    "MatcheringAdvanced": MatcheringAdvanced,
    "MatcheringLimiterConfig": MatcheringLimiterConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Matchering": "Matchering",
    "MatcheringAdvanced": "Matchering (Advanced)",
    "MatcheringLimiterConfig": "Matchering Limiter Config",
}
