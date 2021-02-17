#!/usr/bin/env python3
""" Conditional imports depending on whether the AMD version is installed or not """

from lib.utils import get_backend
from .normalization_common import *

if get_backend() == "amd":
    from .normalization_plaid import *
else:
    from .normalization_tf import *
