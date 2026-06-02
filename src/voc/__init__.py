"""Voice-of-Customer (VoC) intelligence pipeline.

Turns large volumes of unstructured customer text into quantified,
statistically-defensible business insights. See ``docs/SPEC.md``.
"""

from voc.config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
__version__ = "0.1.0"
