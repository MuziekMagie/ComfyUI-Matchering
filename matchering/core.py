# -*- coding: utf-8 -*-

"""
Matchering - Audio Matching and Mastering Python Library
Copyright (C) 2016-2022 Sergree

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .log import Code, info, debug_line, ModuleError
from . import Config
from .stages import main
from .checker import check, check_equality
from .dsp import channel_count, size


def process(
    target_audio,
    reference_audio,
    config: Config = Config(),
):
    debug_line()
    info(Code.INFO_LOADING)

    target = target_audio["waveform"].squeeze(0).numpy().T
    target_sample_rate = target_audio["sample_rate"]

    reference = reference_audio["waveform"].squeeze(0).numpy().T
    reference_sample_rate = reference_audio["sample_rate"]

    # Analyze the target
    target, target_sample_rate = check(target, target_sample_rate, config, "target")

    # Analyze the reference
    reference, reference_sample_rate = check(
        reference, reference_sample_rate, config, "reference"
    )

    # Analyze the target and the reference together
    if not config.allow_equality:
        check_equality(target, reference)

    # Validation of the most important conditions
    if (
        not (target_sample_rate == reference_sample_rate == config.internal_sample_rate)
        or not (channel_count(target) == channel_count(reference) == 2)
        or not (size(target) > config.fft_size and size(reference) > config.fft_size)
    ):
        raise ModuleError(Code.ERROR_VALIDATION)

    # Process
    result, result_no_limiter, result_no_limiter_normalized = main(
        target,
        reference,
        config,
        need_default=True,
        need_no_limiter=True,
        need_no_limiter_normalized=True,
    )

    debug_line()
    info(Code.INFO_COMPLETED)

    return result, result_no_limiter, result_no_limiter_normalized
