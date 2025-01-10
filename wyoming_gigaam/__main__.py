#!/usr/bin/env python3
import argparse
import asyncio
import logging
import types
from functools import partial
from typing import Tuple

import gigaam
import torch
from torch import Tensor
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import GigaAMEventHandler

_LOGGER = logging.getLogger(__name__)


# gigaam.model.LONGFORM_THRESHOLD = 250 * gigaam.preprocess.SAMPLE_RATE


def prepare_wav_fixed(self, buffer: bytearray) -> Tuple[Tensor, Tensor]:
    # pylint: disable=W0212
    wav = torch.frombuffer(buffer, dtype=torch.int16).float() / 32768.0
    wav = wav.to(self._device).to(self._dtype).unsqueeze(0)
    length = torch.full([1], wav.shape[-1], device=self._device)
    return wav, length


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Model to use ('v2_ctc' or 'ctc', 'v2_rnnt' or 'rnnt', 'v1_ctc', 'v1_rnnt')",
        default="rnnt",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference (default: cpu)",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="GigaAM",
                description="Giga Acoustic Model",
                attribution=Attribution(
                    name="SberDevices",
                    url="https://github.com/salute-developers/GigaAM",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name="gigaAM-v1-CTC",
                        description="GigaAM-v1 was trained with a wav2vec2-like approach and was fine-tuned with Connectionist Temporal Classification",
                        attribution=Attribution(
                            name="SberDevices",
                            url="https://github.com/salute-developers/GigaAM",
                        ),
                        installed=args.model == "v1_ctc",
                        languages=["ru"],
                        version="1",
                    ),
                    AsrModel(
                        name="gigaAM-v2-CTC",
                        description="GigaAM-v2 was trained with a HuBERT-like approach and was fine-tuned with Connectionist Temporal Classification",
                        attribution=Attribution(
                            name="SberDevices",
                            url="https://github.com/salute-developers/GigaAM",
                        ),
                        installed=args.model in ("v2_ctc", "ctc"),
                        languages=["ru"],
                        version="1",
                    ),
                    AsrModel(
                        name="GigaAM-v1-RNNT",
                        description="GigaAM-v1 was trained with a wav2vec2-like approach and was fine-tuned with RNN Transducer loss",
                        attribution=Attribution(
                            name="SberDevices",
                            url="https://github.com/salute-developers/GigaAM",
                        ),
                        installed=args.model == "v1_rnnt",
                        languages=["ru"],
                        version="1",
                    ),
                    AsrModel(
                        name="GigaAM-v2-RNNT",
                        description="GigaAM-v2 was trained with a HuBERT-like approach and was fine-tuned with RNN Transducer loss",
                        attribution=Attribution(
                            name="SberDevices",
                            url="https://github.com/salute-developers/GigaAM",
                        ),
                        installed=args.model in ("v2_rnnt", "rnnt"),
                        languages=["ru"],
                        version="1",
                    ),
                ],
            )
        ],
    )
    # Load model
    giga_model = gigaam.load_model(
        model_name=args.model, device=args.device, download_root=args.data_dir
    )

    # redefine giga prepare_wav method to accept bytes instead of using ffmpeg
    giga_model.prepare_wav = types.MethodType(prepare_wav_fixed, giga_model)

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()
    await server.run(
        partial(
            GigaAMEventHandler,
            wyoming_info,
            args,
            giga_model,
            model_lock,
        )
    )


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
