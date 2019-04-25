#!/usr/bin/env python3
""" Memory saving gradients.
Adapted from: https://github.com/openai/gradient-checkpointing

The MIT License

Copyright (c) 2018 OpenAI (http://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import contextlib
import logging
import time
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge  # pylint: disable=no-name-in-module
from toposort import toposort


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
sys.setrecursionlimit(10000)
# refers back to current module if we decide to split helpers out
util = sys.modules[__name__]

# getting rid of "WARNING:tensorflow:VARIABLES collection name is deprecated"
setattr(tf.GraphKeys, "VARIABLES", "variables")

# save original gradients since tf.gradient could be monkey-patched to point
# to our version
from tensorflow.python.ops import gradients as tf_grads_lib  # pylint: disable=no-name-in-module
tf_gradients = tf_grads_lib.gradients

MIN_CHECKPOINT_NODE_SIZE = 1024    # use lower value during testing


# specific versions we can use to do process-wide replacement of tf.gradients
def gradients_speed(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, checkpoints='speed', **kwargs)


def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, checkpoints='memory', **kwargs)


def gradients_collection(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, checkpoints='collection', **kwargs)


def gradients(ys, xs,   # pylint: disable: too-many-statements, too-many-branches
              grad_ys=None, checkpoints='collection', **kwargs):
    '''
    Authors: Tim Salimans & Yaroslav Bulatov

    memory efficient gradient implementation inspired by "Training Deep Nets with Sublinear Memory
    Cost" by Chen et al. 2016 (https://arxiv.org/abs/1604.06174)

    ys,xs,grad_ys,kwargs are the arguments to standard tensorflow tf.gradients
    (https://www.tensorflow.org/versions/r0.12/api_docs/python/train.html#gradients)

    'checkpoints' can either be
        - a list consisting of tensors from the forward pass of the neural net
          that we should re-use when calculating the gradients in the backward pass
          all other tensors that do not appear in this list will be re-computed
        - a string specifying how this list should be determined. currently we support
            - 'speed':  checkpoint all outputs of convolutions and matmuls. these ops are usually
                        the most expensive, so checkpointing them maximizes the running speed
                        (this is a good option if nonlinearities, concats, batchnorms, etc are
                        taking up a lot of memory)
            - 'memory': try to minimize the memory usage
                        (currently using a very simple strategy that identifies a number of
                        bottleneck tensors in the graph to checkpoint)
            - 'collection': look for a tensorflow collection named 'checkpoints', which holds the
                            tensors to checkpoint
    '''

    #    print("Calling memsaving gradients with", checkpoints)
    if not isinstance(ys, list):
        ys = [ys]
    if not isinstance(xs, list):
        xs = [xs]

    bwd_ops = ge.get_backward_walk_ops([y.op for y in ys],
                                       inclusive=True)

    debug_print("bwd_ops: {}".format(bwd_ops))

    # forward ops are all ops that are candidates for recomputation
    fwd_ops = ge.get_forward_walk_ops([x.op for x in xs],
                                      inclusive=True,
                                      within_ops=bwd_ops)
    debug_print("fwd_ops: {}".format(fwd_ops))

    # exclude ops with no inputs
    fwd_ops = [op for op in fwd_ops if op.inputs]

    # don't recompute xs, remove variables
    xs_ops = _to_ops(xs)
    fwd_ops = [op for op in fwd_ops if op not in xs_ops]
    fwd_ops = [op for op in fwd_ops if '/assign' not in op.name]
    fwd_ops = [op for op in fwd_ops if '/Assign' not in op.name]
    fwd_ops = [op for op in fwd_ops if '/read' not in op.name]
    ts_all = ge.filter_ts(fwd_ops, True)  # get the tensors
    ts_all = [t for t in ts_all if '/read' not in t.name]
    ts_all = set(ts_all) - set(xs) - set(ys)

    # construct list of tensors to checkpoint during forward pass, if not
    # given as input
    if type(checkpoints) is not list:
        if checkpoints == 'collection':
            checkpoints = tf.get_collection('checkpoints')

        elif checkpoints == 'speed':
            # checkpoint all expensive ops to maximize running speed
            checkpoints = ge.filter_ts_from_regex(fwd_ops, 'conv2d|Conv|MatMul')

        elif checkpoints == 'memory':

            # remove very small tensors and some weird ops
            def fixdims(t):  # tf.Dimension values are not compatible with int, convert manually
                try:
                    return [int(e if e.value is not None else 64) for e in t]
                except:
                    return [0]  # unknown shape
            ts_all = [t for t in ts_all if np.prod(fixdims(t.shape)) > MIN_CHECKPOINT_NODE_SIZE]
            ts_all = [t for t in ts_all if 'L2Loss' not in t.name]
            ts_all = [t for t in ts_all if 'entropy' not in t.name]
            ts_all = [t for t in ts_all if 'FusedBatchNorm' not in t.name]
            ts_all = [t for t in ts_all if 'Switch' not in t.name]
            ts_all = [t for t in ts_all if 'dropout' not in t.name]
            # DV: FP16_FIX - need to add 'Cast' layer here to make it work for FP16
            ts_all = [t for t in ts_all if 'Cast' not in t.name]

            # filter out all tensors that are inputs of the backward graph
            with util.capture_ops() as bwd_ops:
                tf_gradients(ys, xs, grad_ys, **kwargs)

            bwd_inputs = [t for op in bwd_ops for t in op.inputs]
            # list of tensors in forward graph that is in input to bwd graph
            ts_filtered = list(set(bwd_inputs).intersection(ts_all))
            debug_print("Using tensors {}".format(ts_filtered))

            # try two slightly different ways of getting bottlenecks tensors
            # to checkpoint
            for ts in [ts_filtered, ts_all]:

                # get all bottlenecks in the graph
                bottleneck_ts = []
                for t in ts:
                    b = set(ge.get_backward_walk_ops(t.op, inclusive=True, within_ops=fwd_ops))
                    f = set(ge.get_forward_walk_ops(t.op, inclusive=False, within_ops=fwd_ops))
                    # check that there are not shortcuts
                    b_inp = set([inp for op in b for inp in op.inputs]).intersection(ts_all)
                    f_inp = set([inp for op in f for inp in op.inputs]).intersection(ts_all)
                    if not set(b_inp).intersection(f_inp) and len(b_inp)+len(f_inp) >= len(ts_all):
                        bottleneck_ts.append(t)  # we have a bottleneck!
                    else:
                        debug_print("Rejected bottleneck candidate and ops {}".format(
                            [t] + list(set(ts_all) - set(b_inp) - set(f_inp))))

                # success? or try again without filtering?
                if len(bottleneck_ts) >= np.sqrt(len(ts_filtered)):  # enough bottlenecks found!
                    break

            if not bottleneck_ts:
                raise Exception('unable to find bottleneck tensors! please provide checkpoint '
                                'nodes manually, or use checkpoints="speed".')

            # sort the bottlenecks
            bottlenecks_sorted_lists = tf_toposort(bottleneck_ts, within_ops=fwd_ops)
            sorted_bottlenecks = [t for ts in bottlenecks_sorted_lists for t in ts]

            # save an approximately optimal number ~ sqrt(N)
            N = len(ts_filtered)
            if len(bottleneck_ts) <= np.ceil(np.sqrt(N)):
                checkpoints = sorted_bottlenecks
            else:
                step = int(np.ceil(len(bottleneck_ts) / np.sqrt(N)))
                checkpoints = sorted_bottlenecks[step::step]

        else:
            raise Exception('%s is unsupported input for "checkpoints"' % (checkpoints,))

    checkpoints = list(set(checkpoints).intersection(ts_all))

    # at this point automatic selection happened and checkpoints is list of nodes
    assert isinstance(checkpoints, list)

    debug_print("Checkpoint nodes used: {}".format(checkpoints))
    # better error handling of special cases
    # xs are already handled as checkpoint nodes, so no need to include them
    xs_intersect_checkpoints = set(xs).intersection(set(checkpoints))
    if xs_intersect_checkpoints:
        debug_print("Warning, some input nodes are also checkpoint nodes: {}".format(
            xs_intersect_checkpoints))
    ys_intersect_checkpoints = set(ys).intersection(set(checkpoints))
    debug_print("ys: {}, checkpoints:{}, intersect: {}".format(
        ys, checkpoints, ys_intersect_checkpoints))
    # saving an output node (ys) gives no benefit in memory while creating
    # new edge cases, exclude them
    if ys_intersect_checkpoints:
        debug_print("Warning, some output nodes are also checkpoints nodes: {}".format(
            format_ops(ys_intersect_checkpoints)))

    # remove initial and terminal nodes from checkpoints list if present
    checkpoints = list(set(checkpoints) - set(ys) - set(xs))

    # check that we have some nodes to checkpoint
    if not checkpoints:
        raise Exception('no checkpoints nodes found or given as input! ')

    # disconnect dependencies between checkpointed tensors
    checkpoints_disconnected = {}
    for x in checkpoints:
        if x.op and x.op.name is not None:
            grad_node = tf.stop_gradient(x, name=x.op.name+"_sg")
        else:
            grad_node = tf.stop_gradient(x)
        checkpoints_disconnected[x] = grad_node

    # partial derivatives to the checkpointed tensors and xs
    ops_to_copy = fast_backward_ops(seed_ops=[y.op for y in ys],
                                    stop_at_ts=checkpoints, within_ops=fwd_ops)
    debug_print("Found {} ops to copy within fwd_ops {}, seed {}, stop_at {}".format(
        len(ops_to_copy), fwd_ops, [r.op for r in ys], checkpoints))
    debug_print("ops_to_copy = {}".format(ops_to_copy))
    debug_print("Processing list {}".format(ys))
    _, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
    for origin_op, op in info._transformed_ops.items():
        op._set_device(origin_op.node_def.device)
    copied_ops = info._transformed_ops.values()
    debug_print("Copied {} to {}".format(ops_to_copy, copied_ops))
    ge.reroute_ts(checkpoints_disconnected.values(),
                  checkpoints_disconnected.keys(),
                  can_modify=copied_ops)
    debug_print("Rewired {} in place of {} restricted to {}".format(
        checkpoints_disconnected.values(), checkpoints_disconnected.keys(), copied_ops))

    # get gradients with respect to current boundary + original x's
    copied_ys = [info._transformed_ops[y.op]._outputs[0] for y in ys]
    boundary = list(checkpoints_disconnected.values())
    dv = tf_gradients(ys=copied_ys, xs=boundary+xs, grad_ys=grad_ys, **kwargs)
    debug_print("Got gradients {}".format(dv))
    debug_print("for %s", copied_ys)
    debug_print("with respect to {}".format(boundary+xs))

    inputs_to_do_before = [y.op for y in ys]
    if grad_ys is not None:
        inputs_to_do_before += grad_ys
    wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
    my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

    # partial derivatives to the checkpointed nodes
    # dictionary of "node: backprop" for nodes in the boundary
    d_checkpoints = {r: dr for r, dr in zip(checkpoints_disconnected.keys(),
                                            dv[:len(checkpoints_disconnected)])}
    # partial derivatives to xs (usually the params of the neural net)
    d_xs = dv[len(checkpoints_disconnected):]

    # incorporate derivatives flowing through the checkpointed nodes
    checkpoints_sorted_lists = tf_toposort(checkpoints, within_ops=fwd_ops)
    for ts in checkpoints_sorted_lists[::-1]:
        debug_print("Processing list {}".format(ts))
        checkpoints_other = [r for r in checkpoints if r not in ts]
        checkpoints_disconnected_other = [checkpoints_disconnected[r] for r in checkpoints_other]

        # copy part of the graph below current checkpoint node, stopping at
        # other checkpoints nodes
        ops_to_copy = fast_backward_ops(within_ops=fwd_ops,
                                        seed_ops=[r.op for r in ts],
                                        stop_at_ts=checkpoints_other)
        debug_print("Found {} ops to copy within {}, seed {}, stop_at {}".format(
            len(ops_to_copy), fwd_ops, [r.op for r in ts], checkpoints_other))
        debug_print("ops_to_copy = {}".format(ops_to_copy))
        if not ops_to_copy:  # we're done!
            break
        _, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
        for origin_op, op in info._transformed_ops.items():
            op._set_device(origin_op.node_def.device)
        copied_ops = info._transformed_ops.values()
        debug_print("Copied {} to {}".format(ops_to_copy, copied_ops))
        ge.reroute_ts(checkpoints_disconnected_other, checkpoints_other, can_modify=copied_ops)
        debug_print("Rewired %s in place of %s restricted to %s",
                    checkpoints_disconnected_other, checkpoints_other, copied_ops)

        # gradient flowing through the checkpointed node
        boundary = [info._transformed_ops[r.op]._outputs[0] for r in ts]
        substitute_backprops = [d_checkpoints[r] for r in ts]
        dv = tf_gradients(boundary,
                          checkpoints_disconnected_other+xs,
                          grad_ys=substitute_backprops, **kwargs)
        debug_print("Got gradients {}".format(dv))
        debug_print("for {}".format(boundary))
        debug_print("with respect to {}".format(checkpoints_disconnected_other+xs))
        debug_print("with boundary backprop substitutions {}".format(substitute_backprops))

        inputs_to_do_before = [d_checkpoints[r].op for r in ts]
        wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
        my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

        # partial derivatives to the checkpointed nodes
        for r, dr in zip(checkpoints_other, dv[:len(checkpoints_other)]):
            if dr is not None:
                if d_checkpoints[r] is None:
                    d_checkpoints[r] = dr
                else:
                    d_checkpoints[r] += dr

        def _unsparsify(var_x):
            if not isinstance(var_x, tf.IndexedSlices):
                return var_x
            assert var_x.dense_shape is not None, \
                "memory_saving_gradients encountered sparse gradients of unknown shape"
            indices = var_x.indices
            while indices.shape.ndims < var_x.values.shape.ndims:
                indices = tf.expand_dims(indices, -1)
            return tf.scatter_nd(indices, var_x.values, var_x.dense_shape)

        # partial derivatives to xs (usually the params of the neural net)
        d_xs_new = dv[len(checkpoints_other):]
        for j in range(len(xs)):
            if d_xs_new[j] is not None:
                if d_xs[j] is None:
                    d_xs[j] = _unsparsify(d_xs_new[j])
                else:
                    d_xs[j] += _unsparsify(d_xs_new[j])

    return d_xs


def tf_toposort(ts_inp, within_ops=None):
    """ Tensorflow topological sort """
    all_ops = ge.get_forward_walk_ops([x.op for x in ts_inp], within_ops=within_ops)

    deps = {}
    for tf_op in all_ops:
        for outp in tf_op.outputs:
            deps[outp] = set(tf_op.inputs)
    sorted_ts = toposort(deps)

    # only keep the tensors from our original list
    ts_sorted_lists = []
    for lst in sorted_ts:
        keep = list(set(lst).intersection(ts_inp))
        if keep:
            ts_sorted_lists.append(keep)
    return ts_sorted_lists


def fast_backward_ops(within_ops, seed_ops, stop_at_ts):
    """ Fast backward ops """
    bwd_ops = set(ge.get_backward_walk_ops(seed_ops, stop_at_ts=stop_at_ts))
    ops = bwd_ops.intersection(within_ops).difference([t.op for t in stop_at_ts])
    return list(ops)


@contextlib.contextmanager
def capture_ops():
    """Decorator to capture ops created in the block.
    with capture_ops() as ops:
        # create some ops
    print(ops) # => prints ops created.
    """

    micros = int(time.time()*10**6)
    scope_name = str(micros)
    op_list = []
    with tf.name_scope(scope_name):
        yield op_list

    graph = tf.get_default_graph()
    op_list.extend(ge.select_ops(scope_name+"/.*", graph=graph))


def _to_op(tensor_or_op):
    """ Convert to op """
    if hasattr(tensor_or_op, "op"):
        return tensor_or_op.op
    return tensor_or_op


def _to_ops(iterable):
    """ Convert to ops """
    if not _is_iterable(iterable):
        return iterable
    return [_to_op(i) for i in iterable]


def _is_iterable(obj):
    """ Check if object is iterable """
    try:
        _ = iter(obj)
    except Exception:  # pylint: disable=broad-except
        return False
    return True


def debug_print(msg, *args):
    """ Debug logging """
    formatted_args = [format_ops(arg) for arg in args]
    logger.debug("%s: %s", msg, formatted_args)


def format_ops(ops, sort_outputs=True):
    """Helper method for printing ops. Converts Tensor/Operation op to op.name,
       rest to str(op)."""

    if hasattr(ops, '__iter__') and not isinstance(ops, str):
        lst = [(op.name if hasattr(op, "name") else str(op)) for op in ops]
        if sort_outputs:
            return sorted(lst)
        return lst
    return ops.name if hasattr(ops, "name") else str(ops)


def my_add_control_inputs(wait_to_do_ops, inputs_to_do_before):
    """ Add control inputs """
    for tf_op in wait_to_do_ops:
        ctl_inp = [i for i in inputs_to_do_before
                   if tf_op.control_inputs is None or i not in tf_op.control_inputs]
        ge.add_control_inputs(tf_op, ctl_inp)
