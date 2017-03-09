from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from train import initialize_vocab
from evaluate import exact_match_score, f1_score

FLAGS = tf.app.flags.FLAGS
#
logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

#
class Encoder(object):
    def __init__(self, size, vocab_dim):
        # size = hidden size?
        self.size = size
        self.vocab_dim = vocab_dim
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)

    def encode(self, inputs, masks, encoder_state_input, scope='', reuse=False):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        srclen = tf.reduce_sum(masks, axis=0) # TODO: choose correct axis!
        # (fw_o, bw_o), _ = birectional_dynamic_rnn(self.cell, inputs, srclen=srclen, intial_state=None)

        outputs, state = dynamic_rnn(
            self.cell,
            inputs,
            sequence_length=seclen,
            initial_state=encoder_state_input,
            scope=scope
        )
        return outputs, state #TODO: ????


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, h_q, h_p):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        with vs.scope("answer_start"):
            a_s = rnn_cell._linear([h_q, h_p], output_size=self.output_size)
        with vs.scope("answer_end"):
            a_e = rnn_cell._linear([h_q, h_p], output_size=self.output_size)


        return a_s, a_e

class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        self.question = tf.placeholder(shape=[None, ??????])
        self.paragraph = tf.placeholder(shape=[None, FLAGS.output_size])
        self.start_answer = tf.placeholder(shape=[None, FLAGS.output_size])
        self.end_answer = tf.placeholder(shape=[None, FLAGS.output_size])

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        params = tf.trainable_variables()
        grads = tf.gradient(self.loss, params)

        # "adam" or "sgd"
        self.updates = get_optimizer(FLAGS.optimizer)(FLAGS.learning_rate).minimize(self.loss)
        vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
        self.vocab, self.rev_vocab = initialize_vocab(FLAGS.vocab_path)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encoder = Encoder(FLAGS.state_size, FLAGS.embedding_size)
        q_o, h_q = encoder.encode(input=self.question_var, masks=?????, encoder_state_input=None)
        p_o, h_p = encoder.encode(input=self.paragraph_var, masks=????, encoder_state_input=q_h, reuse=True)

        decoder = Decoder(FLAGS.output_size)
        self.a_s, self.a_e = decoder.decode(h_q, h_p)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """

        with vs.variable_scope("loss"):
            l1 = tf.nn.sparse.softmax_cross_entropy_with_logits(self.a_s, self.start_answer)
            l2 = tf.nn.sparse.softmax_cross_entropy_with_logits(self.a_e, self.end_answer)
            self.loss = l1 + l2

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            glove_matrix = np.load(FLAGS.embed_path)['glove']
            embeddings = tf.Constant(glove_matrix)
            self.paragraph_var = tf.nn.embedding_lookup(embeddings, self.paragraph)
            self.question_var = tf.nn.embedding_lookup(embeddings, self.question)

    def optimize(self, session, paragraph, question, start_answer, end_answer):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {
            self.question: question,
            self.paragraph: paragraph,
            self.start_answer: start_answer,
            self.end_answer: end_answer
        }


        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = [self.updates, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, paragraph, question, start_answer, end_answer):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {
            self.question: question,
            self.paragraph: paragraph,
            self.start_answer: start_answer,
            self.end_answer: end_answer
        }

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = [self.updates, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, paragraph, question):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {
            self.paragraph: parapraph,
            self.question: question
        }

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = [self.a_s, self.a_e]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, paragraph, question):

        yp, yp2 = self.decode(session, paragraph, question)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for paragraph, question, start_answer, end_answer in valid_dataset:
            valid_cost = self.test(sess, paragraph, question, start_answer, end_answer)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=None, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1_values = 0.0
        em_values = 0.0

        if sample:
            sample_indices = random.sample(len(dataset), k=sample)
            dataset = dataset[sample_indices]

        for p, q, ground_truth in dataset:
            a_s, a_e = self.answer(session, p, q)
            prediction = paragraph[a_s: a_e+1]
            f1_values += 1.0 / f1_score(prediction, ground_truth)
            em_values += 1.0 / exact_match_score(prediction, ground_truth)

        f1 = sample / f1_values
        em = sample / em_values

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        def load_dataset(batchsize, *filenames):
            files = [open(f) for f in filenames]
            batch = []
            for i in range(len(files[0])):
                example = []
                for f in files:
                    int_list = [int(x) for x in f.readline().split()]
                    example.append(int_list)
                batch.append(example)

                if len(batch) == batchsize:
                    yield batch
            if len(batch) > 0:
                yield batch

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        for e in range(FLAGS.epochs):
            for p, q, a in dataset:
                # transfer a to be start_ans and end_ans
                loss = self.optimize(session, p, q, start_answer, end_answer)
            ## save the model
            saver = tf.Saver()

            val_loss = self.validate(dataset_train)

            f1_train, em_train = self.evaluate_answer(session, dataset_train, sample=100)
            f1_val, em_val = self.evaluate_answer(session, dataset_test)
