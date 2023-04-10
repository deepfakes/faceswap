#!/usr/bin/env python3
""" PlaidML helper Utilities """
from typing import Optional

import plaidml


def pad(data: plaidml.tile.Value,
        paddings,
        mode: str = "CONSTANT",
        name: Optional[str] = None,  # pylint:disable=unused-argument
        constant_value: int = 0) -> plaidml.tile.Value:
    """ PlaidML Pad

    Notes
    -----
    Currently only Reflect padding is supported.

    Parameters
    ----------
    data :class:`plaidm.tile.Value`
        The tensor to pad
    mode: str, optional
        The padding mode to use. Default: `"CONSTANT"`
    name: str, optional
        The name for the operation. Unused but kept for consistency with tf.pad. Default: ``None``
    constant_value: int, optional
        The value to pad the Tensor with. Default: `0`

    Returns
    -------
    :class:`plaidm.tile.Value`
        The padded tensor
    """
    # TODO: use / implement other padding method when required
    # CONSTANT -> SpatialPadding ? | Doesn't support first and last axis +
    #             no support for constant_value
    # SYMMETRIC -> Requires implement ?
    if mode.upper() != "REFLECT":
        raise NotImplementedError("pad only supports mode == 'REFLECT'")
    if constant_value != 0:
        raise NotImplementedError("pad does not support constant_value != 0")
    return plaidml.op.reflection_padding(data, paddings)


def is_plaidml_error(error: Exception) -> bool:
    """ Test whether the given exception is a plaidml Exception.

    Parameters
    ----------
    error: :class:`Exception`
        The generated error

    Returns
    -------
    bool
        ``True`` if the given error has been generated from plaidML otherwise ``False``
    """
    return isinstance(error, plaidml.exceptions.PlaidMLError)
