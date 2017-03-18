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
from qa_data import PAD_ID

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.10, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 20, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("iteration_size", 4, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("p", 16, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")

tf.app.flags.DEFINE_integer("question_size", 60, "Max Question Length")
tf.app.flags.DEFINE_integer("paragraph_size", 350, "Max Context Paragraph Length")
tf.app.flags.DEFINE_integer("hidden_size", 200, "size of hidden layer h_i")
tf.app.flags.DEFINE_integer("sample_every", 100, "every 100 batch to plot one result")
tf.app.flags.DEFINE_integer("sample_size", 100, "size of sample")


tf.app.flags.DEFINE_integer("perspective_units", 50, "Number of lstm representation h_i")
tf.app.flags.DEFINE_bool("grad_clip", True, "whether or not to clip the gradients")
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "lambda constant for regularization")


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

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    # adapted from keras documentation

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


def load_data(data_dir, data_subset):
    path = data_dir + "/" + data_subset
    #raw answer can be fetched using indices and raw paragraphs
    p, q, p_len, q_len, a_s, a_e, p_raw = ([] for i in range(7))
    with open(path + ".ids.question") as f:
        for line in f:
            q.append(line.split())
            q_len.append(min(FLAGS.question_size, len(line.split())))
        q = pad_sequences(q, maxlen=FLAGS.question_size, value=PAD_ID, padding="post")

    with open(path + ".ids.context") as f:
        for line in f:
            p.append(line.split())
            p_len.append(min(FLAGS.paragraph_size, len(line.split())))
        p = pad_sequences(p, maxlen=FLAGS.paragraph_size, value=PAD_ID, padding="post")

    with open(path + ".span") as f:
        for line in f:
            start_index, end_index = [int(x) for x in line.split()]
            if start_index >= FLAGS.paragraph_size:
                a_len = end_index - start_index + 1
                end_index = FLAGS.paragraph_size -1
                start_index = end_index- a_len + 1
            if start_index < FLAGS.paragraph_size and end_index > FLAGS.paragraph_size - 1:
                end_index = FLAGS.paragraph_size - 1
            a_s.append(start_index)
            a_e.append(end_index)

            assert(start_index < FLAGS.paragraph_size)
            assert(end_index < FLAGS.paragraph_size)
    with open(path + ".context") as f:
        for line in f:
            p_raw.append(line.split())

    return p, q, p_len, q_len, a_s, a_e, p_raw


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

    train_data = zip(*load_data(FLAGS.data_dir, "train"))
    dev_data = zip(*load_data(FLAGS.data_dir, "val"))

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

            qa.train(sess, train_data, dev_data)

            #qa.evaluate_answer(sess, q_dev, p_dev, a_s_dev, vocab)

if __name__ == "__main__":
    tf.app.run()


