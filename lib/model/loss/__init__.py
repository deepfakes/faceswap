#!/usr/bin/env python3
""" Conditional imports depending on whether the AMD version is installed or not """

from lib.utils import get_backend

if get_backend() == "amd":
    from . import loss_plaid as losses # noqa
else:
    from . import loss_tf as losses  # type:ignore # noqa
