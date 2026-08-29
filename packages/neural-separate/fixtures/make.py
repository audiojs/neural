#!/usr/bin/env python3
# Builds the committed test fixture without torch — plain onnx.helper.
#
#   python3 -m venv .venv && .venv/bin/pip install onnx   # numpy comes along
#   .venv/bin/python3 fixtures/make.py
#
# identity-mask.onnx: y = Identity(x). Dynamic batch/channel/bin/frame axes,
# so it accepts any [1, C, F, T] magnitude tensor. Standing in for a trained
# 'openunmix'-modelType target model whose forward pass is the identity
# function — proves the JS-side tensor packing/unpacking and the real
# onnxruntime-node round trip (test 7 in test.js), independent of any actual
# separation quality (which the oracle-mask tests in test.js already cover
# without any ONNX model at all).
import onnx
from onnx import helper, TensorProto

identity = helper.make_node('Identity', ['x'], ['y'], name='identity')

dims = ['batch', 'channels', 'bins', 'frames']
graph = helper.make_graph(
    [identity],
    'identity-mask',
    [helper.make_tensor_value_info('x', TensorProto.FLOAT, dims)],
    [helper.make_tensor_value_info('y', TensorProto.FLOAT, dims)],
)
model = helper.make_model(graph, producer_name='audiojs-neural-separate', opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8  # matches onnxruntime 1.14+ without requiring the newest onnx opset
onnx.checker.check_model(model)
onnx.save(model, 'identity-mask.onnx')

print('wrote identity-mask.onnx (%d bytes)' % len(model.SerializeToString()))
