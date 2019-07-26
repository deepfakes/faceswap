'''
Multiple plaidml implementation.
'''

import plaidml


def pad(data, paddings, mode="CONSTANT", name=None, constant_value=0):
    """ PlaidML Pad """
    # TODO: use / impl other padding method when required
    # CONSTANT -> SpatialPadding ? | Doesn't support first and last axis +
    #             no support for constant_value
    # SYMMETRIC -> Requires impl ?
    if mode.upper() != "REFLECT":
        raise NotImplementedError("pad only supports mode == 'REFLECT'")
    if constant_value != 0:
        raise NotImplementedError("pad does not support constant_value != 0")
    return plaidml.op.reflection_padding(data, paddings)
