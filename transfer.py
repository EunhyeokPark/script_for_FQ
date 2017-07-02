#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./caffe-FQ/python')

from caffe.proto import caffe_pb2
import caffe
import numpy as np

base_proto = "./lenet_train_test.prototxt"
base_weight = "./lenet_iter_10000.caffemodel"
fq_proto = "./lenet_train_test_fq.prototxt"
fq_weight = "./lenet_fq.caffemodel"

active_bit_width = 3
weight_bit_width = 3

# basic functions
def setParam(net, layer_name, index, data):
	layer_idx = list(net._layer_names).index(layer_name)
	np.copyto(net.layers[layer_idx].blobs[index].data, data)

def getBaseParam(proto, weight):
	net = caffe.Net(proto, weight, caffe_pb2.TEST)
	rtn_dict = {}
	for idx, name in enumerate(net._layer_names):
		lst = []
		for blob in net.layers[idx].blobs:
			lst.append(blob.data)			
		if len(lst) > 0:
			rtn_dict[name] = lst			
	return rtn_dict

# load trained parameters
param_dict = getBaseParam(base_proto, base_weight)

# create network with fixed-point layers
net = caffe.Net(fq_proto, caffe_pb2.TRAIN)

# transfer trained parameters to fq net
for idx, name in enumerate(net._layer_names):
	if name in param_dict:
		for b_idx, blob in enumerate(param_dict[name]):
			setParam(net, name, b_idx, blob)
		
		if net.layers[idx].type == "FQConvolution" or \
			net.layers[idx].type == "FQInnerProduct":
			setParam(net, name, len(param_dict[name]), [2**weight_bit_width, 0, 0, 0])
	
	if net.layers[idx].type == "FQActive":
		setParam(net, name, 0, [2**active_bit_width, 0, 0, 0])
		
# run a few epoches to find out proper quantization options.
caffe.set_mode_gpu()
for idx in range(10): 
	net._forward(0, len(net._layer_names)-1)

# store parameters
net.save(fq_weight)

