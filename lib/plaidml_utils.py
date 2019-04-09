'''
Multiple plaidml implementation.
'''
import math

import plaidml
from plaidml.keras import backend as K


class ImagePatches(plaidml.tile.Operation):
    """
    Compatible to tensorflow.extract_image_patches.
    Extract patches from images and put them in the "depth" output dimension.
    Args:
        images: A tensor with a shape of [batch, rows, cols, depth]
        ksizes: The size of the oatches with a shape of [1, patch_rows, patch_cols, 1]
        strides: How far the center of two patches are in the image with a shape
                    of [1, stride_rows, stride_cols, 1]
        rates: How far two consecutive pixel are in the input. Equivalent to dilation.
                Expect shape of [1, rate_rows, rate_cols, 1]
        padding: A string of "VALID" or "SAME" defining padding.
    """
    def __init__(self, images, ksizes, strides, rates=(1, 1, 1, 1), padding="VALID", name=None):
        i_shape = images.shape.dims
        patch_row_eff = ksizes[1] + ((ksizes[1] - 1) * (rates[1] - 1))
        patch_col_eff = ksizes[2] + ((ksizes[2] - 1) * (rates[2] - 1))

        if padding.upper() == "VALID":
            out_rows = math.ceil((i_shape[1] - patch_row_eff + 1.) / float(strides[1]))
            out_cols = math.ceil((i_shape[2] - patch_col_eff + 1.) / float(strides[2]))
            pad_top = 0
            pad_left = 0
        else:
            out_rows = math.ceil(i_shape[1] / float(strides[1]))
            out_cols = math.ceil(i_shape[2] / float(strides[2]))
            pad_top = max(0, ((out_rows - 1) * strides[1] + patch_row_eff - i_shape[1]) // 2)
            pad_left = max(0, ((out_cols - 1) * strides[2] + patch_col_eff - i_shape[2]) // 2)
            # we simply assume padding right == padding left + 1 (same for top/down).
            # This might lead to us padding more as we would need but that won't matter.
            # TF tries to split padding between both sides so pad_left +1 should keep us on the
            # safe side.
            images = K.spatial_2d_padding(images, ((pad_top, pad_top+1), (pad_left, pad_left+1)))

        o_shape = (i_shape[0], out_rows, out_cols, ksizes[1]*ksizes[2]*i_shape[-1])
        code = """function (I[B,Y,X,D]) -> (O) {{
                    TMP[b, ny, nx, y, x, d: B, {NY}, {NX}, {KY}, {KX}, D] =
                        =(I[b, ny * {SY} + y * {RY}, nx * {SX} + x * {RX}, d]);
                    O = reshape(TMP, B, {NY}, {NX}, {KY} * {KX} * D);
                }}
        """.format(NY=out_rows, NX=out_cols,
                   KY=ksizes[1], KX=ksizes[2],
                   SY=strides[1], SX=strides[2],
                   RY=rates[1], RX=rates[2])
        super(ImagePatches, self).__init__(code,
                                           [('I', images), ],
                                           [('O',
                                             plaidml.tile.Shape(images.shape.dtype, o_shape))],
                                           name=name)


extract_image_patches = ImagePatches.function  # pylint: disable=invalid-name


def reflection_padding(inp, paddings):
    """ PlaidML Reflection Padding """
    paddings = [(x, x) if isinstance(x, int) else x for x in paddings]
    ishape = inp.shape.dims
    ndims = inp.shape.ndims
    if len(ishape) != len(paddings):
        raise ValueError("Padding dims != input dims")
    last = inp
    _all_slice = slice(None, None, None)

    def _get_slices(ndims, axis, slice_):
        ret = [_all_slice for _ in range(ndims)]
        ret[axis] = slice_
        return tuple(ret)

    for axis, pads in ((i, x) for i, x in enumerate(paddings) if x[0]+x[1] != 0):
        pad_data = []
        if pads[0]:
            pre = last[_get_slices(ndims, axis, slice(pads[0], 0, -1))]
            pad_data.append(pre)
        pad_data.append(last)
        if pads[1]:
            post = last[_get_slices(ndims, axis, slice(-2, -pads[1]-2, -1))]
            pad_data.append(post)
        last = K.concatenate(pad_data, axis)
        ishape = last.shape.dims
    return last


def pad(data, paddings, mode="CONSTANT", name=None, constant_value=0):
    """ PlaidML Pad """
    # TODO: use / impl other padding method
    # CONSTANT -> SpatialPadding ? | Doesn't support first and last axis +
    #             no support for constant_value
    # SYMMETRIC -> Requires impl ?
    if mode.upper() != "REFLECT":
        raise NotImplementedError("pad only supports mode == 'REFLECT'")
    if constant_value != 0:
        raise NotImplementedError("pad does not support constant_value != 0")
    return reflection_padding(data, paddings)
