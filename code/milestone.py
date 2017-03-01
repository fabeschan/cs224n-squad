from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression

import os, sys

import numpy as np




train_size = 81381
max_length_context = 766
max_length_question = 60
n_classes = 293761
glove_size = 115613


source_dir = os.path.join("data", "squad")

#get list of lists
def loaddata():
	#get train.context

	#train_context, train_question, train_span
	with open(source_dir+'/train.ids.context', 'r') as f:

		train_context = []
		for line in f:
			int_list = [int(x) for x in line.split()]
			train_context.append(int_list)
		#print(train_context[88])

	with open(source_dir+'/train.ids.question', 'r') as f:

		train_question = []
		for line in f:
			int_list = [int(x) for x in line.split()]
			train_question.append(int_list)
		#print(train_question[88])

	with open(source_dir+'/train.span', 'r') as f:
		train_span = []
		for line in f:
			val = [int(x) for x in line.split()]
			#res = int_list[0]*max_length_context + int_list[1]
			res = n_classes - (max_length_context - (val[0]-1))*(max_length_context-(val[0]-1)+1)/2 + (val[1]-val[0]+1)
			train_span.append(res)
		#print(train_span[88])

	return[train_context, train_question, train_span]


def padding(datalist):
	train_context = pad_sequences(datalist[0], maxlen=max_length_context, value=0.)
	train_question = pad_sequences(datalist[1], maxlen=max_length_question, value=0.)

	#print (train_context[88])
	#get train
	return [train_context, train_question]


datas = loaddata()
padded_data = padding(datas)

combined_input = np.concatenate(padded_data, axis = 1)

classed_span = to_categorical(datas[2], nb_classes= n_classes)
print(combined_input.shape)
print(classed_span[88])

#glove = np.load(source_dir+'/glove.trimmed.100.npz')['glove']
#print(glove.shape)

net = input_data(shape=[None, 826])
net = embedding(net, input_dim=115613, output_dim=128)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
net = dropout(net, 0.8)
net = fully_connected(net, n_classes, activation='softmax')
net = regression(net, optimizer='adam', loss='categorical_crossentropy')


model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
model.fit(combined_input, classed_span, validation_set=0.1, show_metric=True, batch_size=64)



























padding(loaddata())