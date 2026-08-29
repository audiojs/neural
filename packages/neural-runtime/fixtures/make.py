#!/usr/bin/env python3
# Builds the two committed test fixtures without torch — plain onnx.helper.
#
#   pip3 install onnx    # numpy comes along as a dependency
#   python3 fixtures/make.py
#
# linear-relu.onnx: y = relu(x @ W + b), fixed weights, input [1, 2] -> output [1, 3].
#   W = [[1, 0, -1], [0.5, -0.5, 2]], b = [0.1, 0, -0.2].
#   Hand check for x = [1, 2]: row @ W = [2.1, -1, 3], +b = [2.1, -1, 2.8], relu = [2.1, 0, 2.8].
#
# scale-shift.onnx: y = x * 2 + 1, dynamic batch dim, input ['batch'] -> output ['batch'].
import numpy as np
import onnx
from onnx import helper, TensorProto

# --- linear-relu.onnx --------------------------------------------------
W = np.array([[1, 0, -1], [0.5, -0.5, 2]], dtype=np.float32)
b = np.array([0.1, 0, -0.2], dtype=np.float32)

gemm = helper.make_node('Gemm', ['x', 'W', 'b'], ['xw'], name='gemm')
relu = helper.make_node('Relu', ['xw'], ['y'], name='relu')

graph = helper.make_graph(
    [gemm, relu],
    'linear-relu',
    [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])],
    [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3])],
    initializer=[
        helper.make_tensor('W', TensorProto.FLOAT, W.shape, W.flatten()),
        helper.make_tensor('b', TensorProto.FLOAT, b.shape, b.flatten()),
    ],
)
model = helper.make_model(graph, producer_name='audiojs-neural-runtime', opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8  # matches onnxruntime 1.14+ without requiring the newest onnx opset
onnx.checker.check_model(model)
onnx.save(model, 'linear-relu.onnx')

# --- scale-shift.onnx ----------------------------------------------------
two = helper.make_tensor('two', TensorProto.FLOAT, [], [2.0])
one = helper.make_tensor('one', TensorProto.FLOAT, [], [1.0])
mul = helper.make_node('Mul', ['x', 'two'], ['xt'], name='mul')
add = helper.make_node('Add', ['xt', 'one'], ['y'], name='add')

graph2 = helper.make_graph(
    [mul, add],
    'scale-shift',
    [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch'])],
    [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch'])],
    initializer=[two, one],
)
model2 = helper.make_model(graph2, producer_name='audiojs-neural-runtime', opset_imports=[helper.make_opsetid('', 13)])
model2.ir_version = 8
onnx.checker.check_model(model2)
onnx.save(model2, 'scale-shift.onnx')

print('wrote linear-relu.onnx (%d bytes), scale-shift.onnx (%d bytes)' % (
    len(model.SerializeToString()), len(model2.SerializeToString())))
