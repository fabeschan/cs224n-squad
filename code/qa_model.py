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
#from train import initialize_vocab
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
class GRUattncell(tf.nn.rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_output, scope=None):
        self.hs = encoder_output
        self.num_units = num_units
        super(GRUattncell, self).__init__(self.num_units)

    def __call__(self, inputs, state, scope=None):
        # for Gru, out and state are the same
        gru_out, gru_state = super(GRUattncell, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self). __name__):
            with vs.variable_scope("attn"):
                ht = tf.nn.rnn_cell._linear(gru_state, self.num_units, True)
                # after expand_dims shape = [num_units, 1]
                # hs = [num_units, num_sources]
                ht = tf.expand_dims(ht, axis=1)
            # scores list length: num_sources
            scores = tf.nn.softmax(tf.reduce_sum(self.hs * ht, axis=0, keep_dims=True))
            context = tf.reduce_sum(self.hs * scores, axis=1)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(tf.nn.rnn_cell._linear([context, gru_state], self.num_units, True))
        return (out, out)

class Encoder(object):
    def __init__(self, size, vocab_dim):
        # size = hidden size?
        self.size = size
        self.vocab_dim = vocab_dim
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        print(self.cell.state_size)

    def encode(self, inputs, masks, scope='', reuse=False):
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
        with vs.variable_scope(scope, reuse):

            seqlen = tf.reduce_sum(masks, axis=1) # TODO: choose correct axis!
            # (fw_o, bw_o), _ = birectional_dynamic_rnn(self.cell, inputs, srclen=srclen, intial_state=None)

            outputs, (f_s, b_s) = tf.nn.bidirectional_dynamic_rnn(
                self.cell,
                self.cell,
                inputs,
                sequence_length=seqlen,
                dtype=tf.float64,
                scope=scope
            )
            # TODO: make sure the out put is all states for using in attention
        return tf.concat(2, outputs), tf.concat(1, [f_s[1], b_s[1]]) #TODO: ????

    def encode_with_attn(self, inputs, masks, prev_states, scope="", reuse=False):
        self.att_gru_cell = GRUattncell(self.size*2, prev_states)
        #self.att_gru_cell = tf.contrib.rnn.AttentionCellWrapper(tf.contrib.rnn.GRUCell, attn_length = FLAGS.question_size, attn_size=None, attn_vec_size=None, input_size=self.size, state_is_tuple=False)
        with vs.variable_scope(scope, reuse):
            seqlen = tf.reduce_sum(masks, axis=1)
            outputs, state = tf.nn.dynamic_rnn(
                self.att_gru_cell,
                inputs,
                sequence_length=seqlen,
                initial_state=None,
                dtype=tf.float64,
                scope=scope
            )
        return outputs, state

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
        with vs.variable_scope("answer_start"):
            a_s = tf.nn.rnn_cell._linear([h_q, h_p], FLAGS.output_size, True)
        with vs.variable_scope("answer_end"):
            a_e = tf.nn.rnn_cell._linear([h_q, h_p], FLAGS.output_size, True)
        return a_s, a_e

def padding_batch(batch, pad_size):
    padded_batch = []
    for p in batch:
        #pad paragraph
        _paragraph, paragraph_mask= padding(pad_size, p)
        padded_batch.append((_paragraph, paragraph_mask))
    return padded_batch

def padding(maxlength, vector):
    original_length = len(vector)
    gap = maxlength - original_length
    if(gap > 0):
        mask = [1]*original_length + [0]*gap
        _vector = vector + gap*[0]
    else:
        mask = [True]*maxlength
        _vector = vector[:maxlength]
    return (_vector, mask)

# def one_hot(examples, size):
#     result = []
#     for i in examples:
#         temp = [0] * size
#         temp[i] = 1
#         result.append(temp)
#     return result

class QASystem(object):
    def __init__(self, encoder, decoder, embed_path, vocab, rev_vocab, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        self.embed_path = embed_path
        self.encoder, self.decoder = encoder, decoder
        self.vocab, self.rev_vocab = vocab, rev_vocab

        # ==== set up placeholder tokens ========
        self.question = tf.placeholder(shape=[None, FLAGS.question_size], dtype=tf.int32)
        self.question_masks = tf.placeholder(shape=[None, FLAGS.question_size], dtype=tf.int32)
        self.paragraph = tf.placeholder(shape=[None, FLAGS.output_size], dtype=tf.int32)
        self.paragraph_masks = tf.placeholder(shape=[None, FLAGS.output_size], dtype=tf.int32)
        self.start_answer = tf.placeholder(shape=[None], dtype=tf.int32)
        self.end_answer = tf.placeholder(shape=[None], dtype=tf.int32)

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        # ==== set up training/updating procedure ====
        params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)

        optimizer = get_optimizer(FLAGS.optimizer)(FLAGS.learning_rate)
        print("grad_clip:",FLAGS.grad_clip)
        if FLAGS.grad_clip:
            clipped_grad, self.norm = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
            optimizer.apply_gradients(zip(clipped_grad, params))
        self.updates = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()

        vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
        #self.vocab, self.rev_vocab = initialize_vocab(FLAGS.vocab_path)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        q_o, h_q = self.encoder.encode(scope='q_enc', inputs=self.question_var, masks=self.question_masks)
        q_o, h_p = self.encoder.encode_with_attn(scope='p_enc', inputs=self.paragraph_var, masks=self.paragraph_masks, prev_states=q_o, reuse=False)
        #p_o, h_p = self.encoder.encode(scope='p_enc', inputs=self.paragraph_var, masks=self.paragraph_masks, encoder_state_input=h_q, reuse=True)

        #self.a_s, self.a_e = self.decoder.decode(h_q[0], h_p[0]) #need double check
        print(h_p)
        self.a_s, self.a_e = self.decoder.decode(h_q, h_p) #need double check


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """

        with vs.variable_scope("loss"):
            l1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.a_s, self.start_answer))
            l2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.a_e, self.end_answer))
            self.loss_s = l1
            self.loss_e = l2
            self.loss = l1 + l2

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('loss_s', self.loss_s)
            tf.summary.scalar('loss_e', self.loss_e)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            glove_matrix = np.load(self.embed_path)['glove']
            embeddings = tf.constant(glove_matrix)
            self.paragraph_var = tf.nn.embedding_lookup(embeddings, self.paragraph)
            self.question_var = tf.nn.embedding_lookup(embeddings, self.question)

    def optimize(self, session, paragraph, question, start_answer, end_answer, paragraph_masks, question_masks):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {
            self.question: question,
            self.paragraph: paragraph,
            self.start_answer: start_answer,
            self.end_answer: end_answer,
            self.paragraph_masks: paragraph_masks,
            self.question_masks: question_masks
        }
        output_feed = [self.merged, self.norm, self.updates, self.loss, self.loss_s, self.loss_e]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def test(self, session, paragraph, question, start_answer, end_answer, paragraph_masks, question_masks):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {
            self.question: question,
            self.paragraph: paragraph,
            self.start_answer: start_answer,
            self.end_answer: end_answer,
            self.paragraph_masks: paragraph_masks,
            self.question_masks: question_masks
        }
        output_feed = [self.loss]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def decode(self, session, paragraph, question, paragraph_masks, question_masks):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {
            self.paragraph: paragraph,
            self.question: question,
            self.paragraph_masks: paragraph_masks,
            self.question_masks: question_masks
        }
        output_feed = [self.a_s, self.a_e]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def answer(self, session, paragraph, question, paragraph_masks, question_masks):
        yp, yp2 = self.decode(session, paragraph, question, paragraph_masks, question_masks)
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
        valid_cost = []
        count = 0
        for batch, num_lines_total in valid_dataset(FLAGS.batch_size):
            count += 1
            p, q, a = zip(*batch)
            start_answer, end_answer = zip(*a)
            # start_answer = one_hot(start_answer, FLAGS.output_size)
            # end_answer = one_hot(end_answer, FLAGS.output_size)
            ### TODO: check the validation output sie and questio_size
            p_pad_mask = padding_batch(p,FLAGS.output_size)
            q_pad_mask = padding_batch(q,FLAGS.question_size)
            p_pad, paragraph_masks = zip(*p_pad_mask)
            q_pad, question_masks = zip(*q_pad_mask)
            valid_cost += self.test(sess, p_pad, q_pad, start_answer, end_answer, paragraph_masks, question_masks)
        mean_cost = sum(valid_cost) / count
        return mean_cost

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

        for batch, num_lines_total in dataset(-1):

            if sample:
                sample_indices = random.sample(range(num_lines_total), k=sample)
                data = [batch[i] for i in sample_indices]

            for example in data:
                p, q, ground_truth = example
                p_pad_mask = padding(FLAGS.output_size, p)
                q_pad_mask = padding(FLAGS.question_size, q)
                p_pad, paragraph_masks = p_pad_mask
                q_pad, question_masks = q_pad_mask
                a_s, a_e = self.answer(session, [p_pad], [q_pad], [paragraph_masks], [question_masks])

                prediction = p_pad[a_s: a_e + 1]
                pred_words = " ".join([self.rev_vocab[i] for i in prediction])
                truth = p_pad[ground_truth[0]: ground_truth[1] + 1]
                truth_words = " ".join([self.rev_vocab[i] for i in truth])

                f1_values += f1_score(pred_words, truth_words)
                em_values += exact_match_score(pred_words, truth_words)

        f1 = f1_values/sample
        em = em_values/sample
        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset_train, dataset_val, train_dir):
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

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        self.writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', session.graph)

        i = 0
        for e in range(FLAGS.epochs):
            print("Epoch {}".format(e))
            for batch, num_lines_total in dataset_train(FLAGS.batch_size):
                if not batch:
                    break
                p, q, a = zip(*batch)
                # transfer a to be start_ans and end_ans
                start_answer, end_answer = zip(*a)
                #start_answer = one_hot(start_answer, FLAGS.output_size)
                #end_answer = one_hot(end_answer, FLAGS.output_size)
                p_pad_mask = padding_batch(p,FLAGS.output_size)
                q_pad_mask = padding_batch(q,FLAGS.question_size)
                p_pad, paragraph_masks = zip(*p_pad_mask)
                q_pad, question_masks = zip(*q_pad_mask)

                summary, norm, opt, loss, loss_s, loss_e = self.optimize(session, p_pad, q_pad, start_answer, end_answer, paragraph_masks, question_masks)
                self.writer.add_summary(summary, i)
                print("loss: {}".format(loss, loss_s, loss_e))
                print("norm: {}".format(norm))
                i += 1

            ## save the model
            #saver = tf.train.Saver()
            self.saver.save(session, FLAGS.train_dir + '/model', global_step=e)

            val_loss = self.validate(session, dataset_val)

            f1_train, em_train = self.evaluate_answer(session, dataset_train, sample=100)
            print('f1_train: {}, em_train: {}'.format(f1_train, em_train))

            f1_val, em_val = self.evaluate_answer(session, dataset_val, sample=100)
            print('f1_val: {}, em_val: {}'.format(f1_val, em_val))



