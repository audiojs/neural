#!/usr/bin/env python3
"""Fixture for test.js, no trained weights needed:

  mean-band.onnx   features [N, 1, bands, frames] -> embedding [N, bands]: each band's mean over
                   time, an encoder-shaped graph to run the matcher end to end

    python fixtures/make.py   (onnx)
"""
from pathlib import Path

HERE = Path(__file__).parent
BANDS, FRAMES = 96, 128


def mean_band():
    import onnx
    from onnx import helper, TensorProto
    b, t = BANDS, FRAMES
    g = helper.make_graph([
        helper.make_node('ReduceMean', ['features', 'axes'], ['m'], keepdims=0),
        helper.make_node('Reshape', ['m', 'shape'], ['embedding']),
    ], 'mean-band',
        [helper.make_tensor_value_info('features', TensorProto.FLOAT, ['n', 1, b, t])],
        [helper.make_tensor_value_info('embedding', TensorProto.FLOAT, ['n', b])],
        [helper.make_tensor('axes', TensorProto.INT64, [1], [3]), helper.make_tensor('shape', TensorProto.INT64, [2], [-1, b])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 18)], producer_name='neural-timbre fixtures')
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, HERE / 'mean-band.onnx')


if __name__ == '__main__':
    mean_band()
