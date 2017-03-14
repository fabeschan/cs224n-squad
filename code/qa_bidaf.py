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

class QASystem(object):
    def __init__(self, embed_path, vocab, rev_vocab):
        """
        Initializes your System

        :param args: pass in more arguments as needed
        """

        self.cell = tf.nn.rnn_cell.GRUCell
        self.embed_path = embed_path
        self.vocab, self.rev_vocab = vocab, rev_vocab
        self.dropout_value = FLAGS.dropout

        # ==== set up placeholder tokens ========
        self.dropout = tf.placeholder(shape=[], dtype=tf.float32)
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
        #grads = tf.gradients(self.loss, params)

        optimizer = get_optimizer(FLAGS.optimizer)(FLAGS.learning_rate)

        grads_and_vars = optimizer.compute_gradients(self.loss, params)
        grads = [x[0] for x in grads_and_vars]
        print("grad_clip:",FLAGS.grad_clip)
        if FLAGS.grad_clip:
            grads, _ = tf.clip_by_global_norm(grads, FLAGS.max_gradient_norm)
            grads_and_vars = zip(grads, params)


        optimizer = optimizer.apply_gradients(grads_and_vars)
        self.norm = tf.global_norm(grads)
        tf.summary.scalar('norm', self.norm)
        self.updates = optimizer

        self.saver = tf.train.Saver()

        vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
        #self.vocab, self.rev_vocab = initialize_vocab(FLAGS.vocab_path)

    def setup_system(self):
        self.initializer = tf.uniform_unit_scaling_initializer(1.0)
        self.setup_context_embed_layer()
        self.setup_attn_layer()
        self.setup_modeling_layer()
        self.setup_final_layer()

    def setup_final_layer():
        # Compute a new vector for each paragraph position that
        # multiplies context-paragraph representation with the attention vector.
        M = tf.concat(2, [self.G, self.M])
        print (M.get_shape())
        M2 = tf.concat(2, [self.G, self.M2])
        print (M2.get_shape())
        with tf.variable_scope("preds_start"):
            self.a_s = tf.squeeze(tf.contrib.layers.fully_connected(M, 1, weights_initializer=self.initializer))
        with tf.variable_scope("preds_end"):
            self.a_e = tf.squeeze(tf.contrib.layers.fully_connected(M2, 1, weights_initializer=self.initializer))

    def setup_context_embed_layer(self):
        # Contextual Embed Layer for question.
        with tf.variable_scope("context_embed"):
            question_cell_fw = self.cell(FLAGS.state_size)
            question_cell_bw = self.cell(FLAGS.state_size)

            seqlen_question = tf.reduce_sum(self.question_masks, axis=1) # TODO: choose correct axis!

            question_outputs, state = tf.nn.bidirectional_dynamic_rnn(
                question_cell_fw,
                question_cell_bw,
                self.question_var,
                sequence_length=seqlen_question,
                dtype=tf.float32,
                scope='embed'
            )
            question_outputs_fw, question_outputs_bw = question_outputs
            state_fw, state_bw = state

            question_outputs = tf.concat(2, question_outputs) # shape = (None, max_question_length, 2*state_size)
            question_outputs = tf.nn.dropout(question_outputs, self.dropout)

            seqlen_paragraph = tf.reduce_sum(self.paragraph_masks, axis=1) # TODO: choose correct axis!

            tf.get_variable_scope().reuse_variables()
            paragraph_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                question_cell_fw,
                question_cell_bw,
                self.paragraph_var,
                sequence_length=seqlen_paragraph,
                initial_state_fw=state_fw,
                initial_state_bw=state_bw,
                dtype=tf.float32,
                scope='embed'
            )
            paragraph_outputs_fw, paragraph_outputs_bw = paragraph_outputs
            paragraph_outputs = tf.concat(2, paragraph_outputs) # shape = (None, max_paragraph_length, 2*state_size)
            paragraph_outputs = tf.nn.dropout(paragraph_outputs, self.dropout)

        self.question_outputs = question_outputs
        self.paragraph_outputs = paragraph_outputs

    # Attention Flow Layer
    def setup_attn_layer(self):
        with tf.variable_scope("attentionlayer"):
            W1 = tf.get_variable("W1", shape = [FLAGS.state_size*2])
            W2 = tf.get_variable("W2", shape = [FLAGS.state_size*2])
            W3 = tf.get_variable("W3", shape = [FLAGS.state_size*2])

        #u => queryvector
        u = tf.expand_dims(self.question_outputs, axis=1)
        #h -> context vector
        h = tf.expand_dims(self.paragraph_outputs, axis=2)
        #Similarity element multiplication
        S = tf.reduce_sum(W1*u*h, axis=3) + tf.reduce_sum(W2*u, axis=3) + tf.reduce_sum(W3*h, axis=3)

        #Context-to-Query attn
        #attention weight
        at = tf.expand_dims(tf.nn.softmax(S), axis =3)
        U_t = tf.reduce_sum(at*u, axis =2)

        #query-to-paragraph attn
        b = tf.nn.softmax(tf.reduce_max(S, axis=2, keep_dims=True))
        h_t = tf.reduce_sum(b*self.paragraph_outputs, axis=1)

        H_tilde = tf.reshape(tf.tile(h_t, [1, FLAGS.output_size]), [-1, FLAGS.output_size, 2*FLAGS.state_size])

        #G
        G = tf.concat(2, [self.paragraph_outputs, U_t, self.paragraph_outputs * U_t, self.paragraph_outputs * H_tilde])
        self.G = G

    def setup_modeling_layer(self):

        with tf.variable_scope("firstmodel"):
            modeling_cell_fw1 = self.cell(FLAGS.state_size)
            modeling_cell_bw1 = self.cell(FLAGS.state_size)

            seqlen_paragraph = tf.reduce_sum(self.paragraph_masks, axis=1) # TODO: choose correct axis!

            outputs1, _ = tf.nn.bidirectional_dynamic_rnn(
                modeling_cell_fw1,
                modeling_cell_bw1,
                self.G,
                sequence_length=seqlen_paragraph,
                dtype=tf.float32,
            )
            M1 = tf.concat(2, outputs1)
            M1 = tf.nn.dropout(M, self.dropout)

        with tf.variable_scope("secondmodel"):
            modeling_cell_fw2 = self.cell(FLAGS.state_size)
            modeling_cell_bw2 = self.cell(FLAGS.state_size)

            seqlen_paragraph = tf.reduce_sum(self.paragraph_masks, axis=1) # TODO: choose correct axis!

            outputs2, _ = tf.nn.bidirectional_dynamic_rnn(
                modeling_cell_fw2,
                modeling_cell_bw2,
                M1,
                sequence_length=seqlen_paragraph,
                dtype=tf.float32,
            )
            M2 = tf.concat(2, outputs2) # shape = (None, max_paragraph_length, 2*state_size)
            M2 = tf.nn.dropout(M2, self.dropout)
        self.M1 = M1
        self.M2 = M2


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
            glove_matrix = np.load(self.embed_path)['glove'].astype(np.float32)
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
            self.dropout: FLAGS.dropout,
            self.question: question,
            self.paragraph: paragraph,
            self.start_answer: start_answer,
            self.end_answer: end_answer,
            self.paragraph_masks: paragraph_masks,
            self.question_masks: question_masks
        }
        output_feed = [self.merged, self.norm, self.updates, self.loss]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def test(self, session, paragraph, question, start_answer, end_answer, paragraph_masks, question_masks):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {
            self.dropout: FLAGS.dropout,
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
            self.dropout: FLAGS.dropout,
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
                start_answer, end_answer = zip(*a)
                p_pad_mask = padding_batch(p,FLAGS.output_size)
                q_pad_mask = padding_batch(q,FLAGS.question_size)
                p_pad, paragraph_masks = zip(*p_pad_mask)
                q_pad, question_masks = zip(*q_pad_mask)

                summary, norm, opt, loss = self.optimize(session, p_pad, q_pad, start_answer, end_answer, paragraph_masks, question_masks)
                self.writer.add_summary(summary, i)
                print("loss: {}".format(loss))
                if (norm > FLAGS.max_gradient_norm):
                    print("boom, I exploded here")
                print("norm: {}".format(norm))
                i += 1

            ## save the model
            self.saver.save(session, FLAGS.train_dir + '/model', global_step=e)

            val_loss = self.validate(session, dataset_val)

            f1_train, em_train = self.evaluate_answer(session, dataset_train, sample=100)
            print('f1_train: {}, em_train: {}'.format(f1_train, em_train))

            f1_val, em_val = self.evaluate_answer(session, dataset_val, sample=100)
            print('f1_val: {}, em_val: {}'.format(f1_val, em_val))



