# LSTM and CNN for sequence classification in the IMDB dataset
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

# fix random seed for reproducibility
np.random.seed(7)

'''
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# ========================================================
'''

train_size = 81381
max_length_context = 766
max_length_question = 60
max_x_length = max_length_context + max_length_question
#n_classes = 293761
n_classes = 2
glove_size = 115613

source_dir = os.path.join("data", "squad")

def loaddata():
    with open(source_dir + '/train.ids.context', 'r') as f:
        train_context = []
        train_context_lengthlist = []
        for line in f:
            int_list = [int(x) for x in line.split()]
            train_context_lengthlist.append(len(int_list))
            train_context.append(int_list)

    with open(source_dir + '/train.ids.question', 'r') as f:
        train_question = []
        train_question_lengthlist = []
        for line in f:
            int_list = [int(x) for x in line.split()]
            train_question_lengthlist.append(len(int_list))
            train_question.append(int_list)

    with open(source_dir + '/train.span', 'r') as f:
        train_span = []
        train_span_lengthlist = []
        count = 0
        for line in f:
            val = [int(x) for x in line.split()]
            train_span_lengthlist.append(val[1] - val[0] + 1)
            res = np.zeros(train_context_lengthlist[count])
            res[val[0]:val[1] + 1]  = 1
            train_span.append(res)
            count += 1
    return[train_context, train_question, train_span, train_context_lengthlist, train_question_lengthlist, train_span_lengthlist]



def padding(datalist):
    train_context = sequence.pad_sequences(datalist[0], maxlen=max_length_context, value=0.)
    train_question = sequence.pad_sequences(datalist[1], maxlen=max_length_question, value=0.)
    train_span = sequence.pad_sequences(datalist[2], maxlen=max_length_context, value=0.)
    return [train_context, train_question, train_span]


if __name__ == '__main__':
    datas = loaddata()
    padded_data = padding(datas)

    '''
    try to plot the histogram of length
    '''
    train_context_lengthlist = datas[3]
    plt.hist(train_context_lengthlist)
    #plt.axis([0, 42, 50000, 85000])
    plt.xlabel('train_context_lengthlist')
    plt.ylabel('Frequency')
    plt.title('Train_context length Histogram')
    plt.grid(True)
    output_path = "{}.png".format("context")
    plt.savefig(output_path)

    train_question_lengthlist = datas[4]
    plt.hist(train_question_lengthlist)
    #plt.axis([0, 42, 50000, 85000])
    plt.xlabel('train_question_lengthlist')
    plt.ylabel('Frequency')
    plt.title('Train_question length Histogram')
    plt.grid(True)
    output_path = "{}.png".format("question")
    plt.savefig(output_path)

    train_span_lengthlist = datas[5]
    plt.hist(train_span_lengthlist)
    #plt.axis([0, 42, 50000, 85000])
    plt.xlabel('train_span_lengthlist')
    plt.ylabel('Frequency')
    plt.title('Train_span length Histogram')
    plt.grid(True)
    output_path = "{}.png".format("span")
    plt.savefig(output_path)

    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(input_dim=glove_size, output_dim=128, input_length=max_length_question))
    model.add(LSTM(100))
    model.add(Dense(output_dim=max_length_context, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    #combined_input = np.concatenate(padded_data, axis = 1)
    #now combined_input includes only questions
    combined_input = padded_data[1]
    model.fit(combined_input, padded_data[2], nb_epoch=10, batch_size=128, validation_split=0.1)
    
    '''
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # ===================================

    combined_input = np.concatenate(padded_data, axis = 1)

    classed_span = to_categorical(datas[2], nb_classes= n_classes)
    print(combined_input.shape)
    print(classed_span[88])

    #glove = np.load(source_dir+'/glove.trimmed.100.npz')['glove']
    #print(glove.shape)

    net = input_data(shape=[None, max_x_length])
    net = embedding(net, input_dim=115613, output_dim=128)
    net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
    net = dropout(net, 0.8)
    net = fully_connected(net, n_classes, activation='softmax')
    net = regression(net, optimizer='adam', loss='categorical_crossentropy')
    print('hiii')


    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
    model.fit(combined_input, classed_span, validation_set=0.1, show_metric=True, batch_size=64)
    '''

