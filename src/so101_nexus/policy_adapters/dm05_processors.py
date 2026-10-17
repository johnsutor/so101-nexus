"""LeRobot tokenization for the original DM0.5 SO101 LoRA release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.policies.dm05.core.tokenization import DM05Tokenization
from lerobot.policies.dm05.processor_dm05 import DM05TokenizerProcessorStep
from lerobot.processor import ProcessorStepRegistry


class _SO101Tokenization(DM05Tokenization):
    def _build_user_content(self, **kwargs: Any) -> list:
        # The raw SO101 release was trained with the embodiment in every prompt.
        kwargs["meta_data"] = {**kwargs["meta_data"], "robot_type": "SO101"}
        return super()._build_user_content(**kwargs)


@ProcessorStepRegistry.register(name="dm05_so101_tokenizer_processor")
@dataclass
class DM05SO101TokenizerProcessorStep(DM05TokenizerProcessorStep):
    """Preserve the raw SO101 release's prompt with LeRobot image processing."""

    @property
    def tokenization(self) -> DM05Tokenization:
        """Load the native tokenizer with the original SO101 prompt metadata."""
        if self._tokenization is None:
            self._tokenization = _SO101Tokenization(
                processor=self._load_processor(),
                max_length=self.tokenizer_max_length,
                add_state=self.add_state,
            )
        return self._tokenization
