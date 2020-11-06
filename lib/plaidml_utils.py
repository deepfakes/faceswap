'''
Multiple plaidml implementation.
'''

import plaidml


def pad(data, paddings, mode="CONSTANT", name=None, constant_value=0):
    """ PlaidML Pad """
    # TODO: use / implement other padding method when required
    # CONSTANT -> SpatialPadding ? | Doesn't support first and last axis +
    #             no support for constant_value
    # SYMMETRIC -> Requires implement ?
    if mode.upper() != "REFLECT":
        raise NotImplementedError("pad only supports mode == 'REFLECT'")
    if constant_value != 0:
        raise NotImplementedError("pad does not support constant_value != 0")
    return plaidml.op.reflection_padding(data, paddings)


def is_plaidml_error(error):
    """ Test whether the given exception is a plaidml Exception.

    error: :class:`Exception`
        The generated error

    Returns
    -------
    bool
        ``True`` if the given error has been generated from plaidML otherwise ``False``
    """
    return isinstance(error, plaidml.exceptions.PlaidMLError)
