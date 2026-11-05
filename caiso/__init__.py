"""
optimas
~~~~~~~~~
Package initialisation: version metadata, global configuration flags
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Public package metadata                                                     #
# --------------------------------------------------------------------------- #
__version__ = "0.1.0"         
__author__  = "CAISO contributors; benchmark framework adapted from the Optimas team"

__all__ = [
    "__version__",
    "ENABLE_OPTIMAS_LOGGING",
    "RUN_USER_DIR",
]

# --------------------------------------------------------------------------- #
# Utility: boolean env-var parser                                             #
# --------------------------------------------------------------------------- #
def _env_flag(name: str, default: str = "1") -> bool:
    """
    Convert an environment variable to bool.

    Truthy strings : "1", "true", "yes", "on"   (case-insensitive)
    Falsy  strings : "0", "false", "no", "off"
    """
    value = os.getenv(name, default).lower()
    if value not in {"1", "true", "yes", "on", "y", "t", "0", "false", "no", "off", "n", "f"}:
        value = default.lower()
    return value in {"1", "true", "yes", "on", "y", "t"}


# --------------------------------------------------------------------------- #
# Global logging switch                                                       #
# --------------------------------------------------------------------------- #
ENABLE_OPTIMAS_LOGGING: bool = _env_flag("ENABLE_OPTIMAS_LOGGING", "1")


_pkg_logger = logging.getLogger("caiso")

if ENABLE_OPTIMAS_LOGGING:
    # Configure basic console output if the user hasn’t configured logging yet.
    # We guard with "if not root.handlers" to avoid double-configuration.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    _pkg_logger.info("CAISO logging enabled.")
else:
    # Silence *all* log records emitted from optimas.* by:
    # 1) setting a level higher than CRITICAL
    # 2) preventing propagation to the root logger
    # 3) attaching a NullHandler
    _pkg_logger.setLevel(logging.CRITICAL)
    _pkg_logger.propagate = False
    _pkg_logger.handlers.clear()
    _pkg_logger.addHandler(logging.NullHandler())


# --------------------------------------------------------------------------- #
# LiteLLM                                                                     #
# --------------------------------------------------------------------------- #
import litellm
from caiso.azure_env import install_azure_routing

install_azure_routing()

# Diable the cache by default, as it is not needed in most cases.
litellm.disable_cache()

# Also silence the LiteLLM logger.
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
_pkg_logger.info("Disable LiteLLM cache and logging by default. ")

from caiso.compat import enable_legacy_pandas_pickle

enable_legacy_pandas_pickle()
