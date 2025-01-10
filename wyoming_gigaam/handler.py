"""Event handler for clients of the server."""
import argparse
import asyncio
import logging
from typing import Union

from gigaam import GigaAM, GigaAMASR, GigaAMEmo
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


class GigaAMEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: Union[GigaAM, GigaAMEmo, GigaAMASR],
        model_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.audio_buffer: bytearray | None = None

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self.audio_buffer is None:
                self.audio_buffer = bytearray()

            self.audio_buffer.extend(chunk.audio)

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped. Transcribing...")
            assert self.audio_buffer is not None

            async with self.model_lock:
                transcription = self.model.transcribe(self.audio_buffer)

            self.audio_buffer = None

            _LOGGER.info(transcription)

            await self.write_event(Transcript(text=transcription).event())
            _LOGGER.debug("Completed request")

            # Reset
            return False

        if Transcribe.is_type(event.type):
            Transcribe.from_event(event)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
