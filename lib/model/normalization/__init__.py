#!/usr/bin/env python3
""" Conditional imports depending on whether the AMD version is installed or not """

from lib.utils import get_backend
from .normalization_common import AdaInstanceNormalization
from .normalization_common import GroupNormalization
from .normalization_common import InstanceNormalization


if get_backend() == "amd":
    from .normalization_plaid import LayerNormalization, RMSNormalization
else:
    from .normalization_tf import LayerNormalization, RMSNormalization
