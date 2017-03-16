from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from qa_model import QASystem
from os.path import join as pjoin
import numpy as np

# for load, pad data
from reader import load_data
#from keras.preprocessing.sequence import pad_sequences
#avoid keras = > causing GPU problem

from qa_data import PAD_ID


import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.1, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 20, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 20, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data0/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")

tf.app.flags.DEFINE_integer("QMAXLEN", 60, "Max Question Length")
tf.app.flags.DEFINE_integer("PMAXLEN", 766, "Max Context Paragraph Length")
tf.app.flags.DEFINE_integer("hidden_size", 200, "size of hidden layer h_i")
tf.app.flags.DEFINE_integer("perspective_units", 50, "Number of lstm representation h_i")
tf.app.flags.DEFINE_bool("clip_gradients", True, "Do gradient clipping")
tf.app.flags.DEFINE_bool("tiny_sample", False, "Work with tiny sample")
tf.app.flags.DEFINE_float("tiny_sample_pct", 0.1, "Sample pct.")
tf.app.flags.DEFINE_float("dev_tiny_sample_pct", 1, "Sample pct.")
tf.app.flags.DEFINE_float("span_l2", 0.0001, "Span l2 loss regularization constant")



FLAGS = tf.app.flags.FLAGS

# Define globals here
def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        tf.train.Saver().restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        print(tf.trainable_variables())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):

    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def get_mask(vectors, max_length):
    res = []
    for vector in vectors:
        trulen = len(vector)
        padlen = max_length - trulen
        res.append([1]*trulen + [0]*padlen)
    return pad_sequences(res, maxlen=max_length, value=0, padding="post" )

def load_dataset(*filenames):
    num_lines = sum(1 for line in open(filenames[-1]))
    def generate_batch(batchsize):
        files = [open(f) for f in filenames]
        batch = []
        for i in range(num_lines):
            example = []
            for f in files:
                int_list = [int(x) for x in f.readline().split()]
                example.append(int_list)
            batch.append(example)
            if batchsize != -1:
                if len(batch) == batchsize:
                    yield batch, num_lines
                    #return # for speed
                    batch = []
        if len(batch) > 0:
            yield batch, num_lines
    return generate_batch

def main(_):
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)
    logger = logging.getLogger()

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = None

    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)


    # load data sets
    Q_train, P_train, A_start_train, A_end_train = load_data(FLAGS.data_dir, "val")
    Q_dev, P_dev, A_start_dev, A_end_dev = load_data(FLAGS.data_dir, "val")
    #Q_test, P_test, A_start_test, A_end_test = load_data(FLAGS.data_dir, "test")

    # see some data
    logger.info("Training samples read... %s" % (len(Q_train)))
    logger.info("Dev samples read... %s" % (len(Q_dev)))
    # logger.info("Before Padding: \n Q_train[0]: %s \n P_train[0]: %s \n A_start_train[0]: %s \n A_end_train[0]: %s" % (Q_train[0], P_train[0], A_start_train[0], A_end_train[0]))

    # pad the data at load-time. So, we don't need to do any masking later!!!
    # ref: https://keras.io/preprocessing/sequence/
    # if len < maxlen, pad with specified val
    # elif len > maxlen, truncate
    QMAXLEN = FLAGS.QMAXLEN
    PMAXLEN = FLAGS.PMAXLEN
    Q_mask_train = get_mask(Q_train, QMAXLEN)
    P_mask_train = get_mask(P_train, PMAXLEN)
    Q_train = pad_sequences(Q_train, maxlen=QMAXLEN, value=PAD_ID, padding="post")
    P_train = pad_sequences(P_train, maxlen=PMAXLEN, value=PAD_ID, padding="post")
    train_data = zip(P_train, Q_train, A_start_train, A_end_train, Q_mask_train, P_mask_train)

    # see the effect of padding
    # logger.info("After Padding: \n Q_train[0]: %s \n P_train[0]: %s \n A_start_train[0]: %s \n A_end_train[0]: %s" % (Q_train[0], P_train[0], A_start_train[0], A_end_train[0]))
    # repeat on dev and test set
    Q_mask_dev = get_mask(Q_dev, QMAXLEN)
    P_mask_dev = get_mask(P_dev, PMAXLEN)
    Q_dev = pad_sequences(Q_dev, maxlen=QMAXLEN, value=PAD_ID, padding="post")
    P_dev = pad_sequences(P_dev, maxlen=PMAXLEN, value=PAD_ID, padding="post")
    dev_data = zip(P_dev, Q_dev, A_start_dev, A_end_dev, Q_mask_dev, P_mask_dev)


    global_train_dir = '/tmp/cs224n-squad-train'
    # Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    # file paths saved in the checkpoint. This allows the model to be reloaded even
    # if the location of the checkpoint files has moved, allowing usage with CodaLab.
    # This must be done on both train.py and qa_answer.py in order to work.
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    os.symlink(os.path.abspath(FLAGS.train_dir), global_train_dir)
    train_dir = global_train_dir


    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            logger.info("Loading embeddings")
            embeddings = np.load(FLAGS.data_dir + '/glove.trimmed.' + str(FLAGS.embedding_size) + '.npz')
            pretrained_embeddings = embeddings['glove']
            logger.info("Embeddings loaded with shape: %s %s" % (pretrained_embeddings.shape))

            qa = QASystem(FLAGS, pretrained_embeddings, vocab_dim=len(vocab.keys()))

            initialize_model(sess, qa, train_dir)

            # a reasonable model should perhaps give decent results (f1 in double digits) even with training on smaller set of train_data
            if FLAGS.tiny_sample:
                sample_pct = FLAGS.tiny_sample_pct # sample sample_pct % from train and test for local dev
                sam_train =  np.random.choice(range(len(train_data)), int(sample_pct/100*len(train_data)))
                # no need to sample dev
                sam_dev =  range(len(dev_data)) #np.random.choice(range(len(dev_data)), int(FLAGS.dev_tiny_sample_pct/100*len(dev_data)))
                # small sample
                train_data = [train_data[i] for i in sam_train]
                dev_data = [dev_data[i] for i in sam_dev]

            qa.train(sess, train_data, dev_data)

            #qa.evaluate_answer(sess, Q_dev, P_dev, A_start_dev, vocab)

if __name__ == "__main__":
    tf.app.run()
