#!/usr/bin/env python3
""" Conditional imports depending on whether the AMD version is installed or not """

from lib.utils import get_backend

if get_backend() == "amd":
    from . import losses_plaid as losses
else:
    from . import losses_tf as losses
