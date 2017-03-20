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
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest

import os
from os.path import join as pjoin

from evaluate import exact_match_score, f1_score
from evaluate import evaluate

FLAGS = tf.app.flags.FLAGS

# Setup logging (to console and to file)
logging.root.handlers = []
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO , filename='info.log')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

class Timer(object):
    def __init__(self, name="unnamed"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logging.info("Timer: [{}] took {}s to run".format(self.name, self.interval))

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def batch_linear(args, output_size, bias, bias_start=0.0, scope=None, name=None):
  """Linear map: concat(W[i] * args[i]), where W[i] is a variable.
  Args:
    args: a 3D Tensor with shape [batch x m x n].
    output_size: int, second dimension of W[i] with shape [output_size x m].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.
    name: (optional) variable name.
  Returns:
    A 3D Tensor with shape [batch x output_size x n] equal to
    concat(W[i] * args[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if args.get_shape().ndims != 3:
    raise ValueError("`args` must be a 3D Tensor")

  shape = args.get_shape()
  m = shape[1].value
  n = shape[2].value
  dtype = args.dtype

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    w_name = "weights_"
    if name is not None: w_name += name
    weights = vs.get_variable(
        w_name, [output_size, m], dtype=dtype)
    res = tf.map_fn(lambda x: math_ops.matmul(weights, x), args)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      b_name = "biases_"
      if name is not None: b_name += name
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          b_name, [output_size, n],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
  return tf.map_fn(lambda x: math_ops.add(x, biases), res)

def get_minibatches(data, batch_size=-1, shuffle=True):
    batch = []
    indices = range(len(data))
    if shuffle:
        random.shuffle(indices)
    for i in indices:
        batch.append(data[i])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch):
        yield batch

class QASystem(object):
    def __init__(self, FLAGS, pretrained_embeddings, vocab_dim, *args):
        self.vocab_dim = vocab_dim
        self.pretrained_embeddings = pretrained_embeddings
        self.initializer = tf.contrib.layers.xavier_initializer()

        with Timer("setup graph"):
            with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
                self.setup_placeholder()
                self.setup_system()
                self.setup_loss()
                self.setup_optimizer()
        self.saver = tf.train.Saver()

    def setup_attention_layer(self, D, Q):
        L = tf.matmul(tf.transpose(D, perm=[0, 2, 1]), Q) # L shape: (?, paragraph_size+1, question_size+1)
        A_Q = tf.nn.softmax(L, dim=1) # L shape: (?, paragraph_size+1, question_size+1)
        A_D = tf.nn.softmax(tf.transpose(L, perm=[0,2,1]), dim=1)
        C_Q = tf.matmul(D, A_Q) # C_Q shape (?, hidden_size, question_size+1)
        C_D = tf.matmul(tf.concat([Q, C_Q], 1), A_D) # C_D shape: (?, hidden_size*2, paragraph_size+1)s
        coatt_layer = tf.concat([D, C_D], 1)
        return coatt_layer

    def setup_placeholder(self):
        self.dropout = tf.placeholder(tf.float32, shape=())
        self.q = tf.placeholder(tf.int32, [None, FLAGS.question_size], name="question")
        self.p = tf.placeholder(tf.int32, [None, FLAGS.paragraph_size], name="paragraph")
        self.a_s = tf.placeholder(tf.int32, [None], name="answer_start")
        self.a_e = tf.placeholder(tf.int32, [None], name="answer_end")
        self.p_len = tf.placeholder(tf.int32, [None], name="paragraph_len")
        self.q_len = tf.placeholder(tf.int32, [None], name = "question_len")

    def setup_optimizer(self):
        optimizer = get_optimizer(FLAGS.optimizer)(FLAGS.learning_rate) #.minimize(self.loss)
        variables = tf.trainable_variables()
        gradients = optimizer.compute_gradients(self.loss, variables)
        gradients = [tup[0] for tup in gradients]
        if FLAGS.grad_clip:
            gradients, norms = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.norm = tf.global_norm(gradients)
        grads_and_vars = zip(gradients, variables)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def setup_system(self):
        self.cell = tf.contrib.rnn.BasicLSTMCell

        self.embedding_mat = tf.Variable(self.pretrained_embeddings, name="embed", dtype=tf.float32, trainable=False)
        self.q_emb = tf.nn.embedding_lookup(self.embedding_mat, self.q)
        self.p_emb = tf.nn.embedding_lookup(self.embedding_mat, self.p)

        with tf.variable_scope("encode_qp"):
            cell_fwd = self.cell(FLAGS.hidden_size)
            cell_bwd = self.cell(FLAGS.hidden_size)

            (output_q_fw, output_q_bw), (output_state_fw_q, output_state_bw_q) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fwd,
                cell_bw=cell_bwd,
                inputs=self.q_emb,
                sequence_length=self.q_len,
                dtype=tf.float32,
                scope='encode'
            )
            self.qq = tf.concat([output_q_fw, output_q_bw], 2)
            self.qq = tf.nn.dropout(self.qq, self.dropout)
            fn_add_column = lambda x: tf.concat([x, tf.zeros([1, FLAGS.hidden_size*2], dtype=tf.float32)], 0)
            self.qq = tf.map_fn(lambda x: fn_add_column(x), self.qq, dtype=tf.float32)
            self.qq = tf.tanh(batch_linear(self.qq, FLAGS.question_size+1, True))
            self.qq = tf.transpose(self.qq, perm=[0, 2, 1])

            tf.get_variable_scope().reuse_variables()
            outputs_p, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fwd,
                cell_bw=cell_bwd,
                initial_state_fw=output_state_fw_q,
                initial_state_bw=output_state_bw_q,
                inputs=self.p_emb,
                sequence_length=self.p_len,
                dtype=tf.float32,
                scope='encode'
            )
            self.pp = tf.concat(outputs_p, 2)
            self.pp = tf.nn.dropout(self.pp, self.dropout)
            fn_add_column = lambda x: tf.concat([x, tf.zeros([1, FLAGS.hidden_size*2], dtype=tf.float32)], 0)
            self.pp = tf.map_fn(lambda x: fn_add_column(x), self.pp, dtype=tf.float32)
            self.pp = tf.transpose(self.pp, perm=[0, 2, 1])

        self.coatt_layer = self.setup_attention_layer(self.pp, self.qq)
        G = tf.transpose(self.coatt_layer, perm = [0, 2, 1])
        logging.info('G shape: {}'.format(G.get_shape()))

        with tf.variable_scope("mod", initializer=self.initializer):
            cell_fw = self.cell(FLAGS.hidden_size)
            cell_bw = self.cell(FLAGS.hidden_size)

            output_m, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                G,
                sequence_length=self.p_len,
                dtype=tf.float32
            )
            M = tf.concat(output_m, 2)
            M = tf.nn.dropout(M, self.dropout)

        with tf.variable_scope("mod2", initializer=self.initializer):
            cell_fw = self.cell(FLAGS.hidden_size)
            cell_bw = self.cell(FLAGS.hidden_size)

            output_m2, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                M,
                sequence_length=self.p_len,
                dtype=tf.float32
            )
            M2 = tf.concat(output_m2, 2)
            M2 = tf.nn.dropout(M2, self.dropout)

        M = tf.concat([G, M], 2)
        M2 = tf.concat([G, M2], 2)
        with tf.variable_scope("yp_start", initializer=self.initializer):
            self.logits_start_1 = tf.squeeze(tf.contrib.layers.fully_connected(M, 1, weights_initializer=self.initializer))
            logging.info('logits_start_1 shape: {}'.format(self.logits_start_1.get_shape()))
            self.logits_start = tf.nn.softmax(self.logits_start_1, -1)
            self.yp_start = self.logits_start

        with tf.variable_scope("yp_end", initializer=self.initializer):
            self.logits_end_1 = tf.squeeze(tf.contrib.layers.fully_connected(M2, 1, weights_initializer=self.initializer))
            logging.info('logits_end_1 shape: {}'.format(self.logits_end_1.get_shape()))
            self.logits_end = tf.nn.softmax(self.logits_end_1, -1)
            self.yp_end = self.logits_end

        '''
        cell_final_fw = self.cell(FLAGS.hidden_size*2, state_is_tuple=True)
        cell_final_bw = self.cell(FLAGS.hidden_size*2, state_is_tuple=True)
        U, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_final_fw,
            cell_bw=cell_final_bw,
            inputs=tf.transpose(self.coatt_layer, perm = [0, 2, 1]),
            sequence_length=self.p_len ,
            dtype=tf.float32
        )
        U = tf.concat(U, 2)
        U = tf.nn.dropout(U, self.dropout)
        U_reshape = tf.reshape(U, shape=[-1,FLAGS.hidden_size*4])

        # softmax layer
        W_start = tf.get_variable("W_start", shape=[4*FLAGS.hidden_size, 1], initializer=self.initializer)
        b_start = tf.Variable(tf.zeros([FLAGS.paragraph_size]))
        self.logits_start_1 = tf.reshape(tf.matmul(U_reshape, W_start), shape=[-1, FLAGS.paragraph_size]) + b_start
        self.logits_start = tf.nn.softmax(self.logits_start_1, -1)
        self.yp_start = self.logits_start
        #self.U_s_index = tf.argmax(self.logits_start,1)

        W_end = tf.get_variable("W_end", shape=[4*FLAGS.hidden_size, 1], initializer=self.initializer)
        b_end = tf.Variable(tf.zeros([FLAGS.paragraph_size]))
        self.logits_end_1 = (tf.reshape(tf.matmul(U_reshape, W_end), shape=[-1, FLAGS.paragraph_size]) + b_end)
        self.logits_end = tf.nn.softmax(self.logits_end_1, -1)
        #self.U_e_index = tf.argmax(self.logits_end,1)
        self.yp_end = self.logits_end
        '''

    def setup_loss(self):
        with vs.variable_scope("loss"):
            self.loss_start_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start_1, labels=self.a_s))
            self.loss_end_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end_1, labels=self.a_e))
            a_s_pred = tf.argmax(self.yp_start, axis=1)
            a_e_pred = tf.argmax(self.yp_end, axis=1)
            self.loss_span = tf.reduce_mean(tf.nn.l2_loss(tf.cast(self.a_e - self.a_s + 1, tf.float32) - tf.cast(a_e_pred - a_s_pred + 1, tf.float32)))
            self.loss = tf.add(self.loss_start_1, self.loss_end_1) + FLAGS.l2_lambda * self.loss_span

    def evaluate_answer(self, session, p, q, p_len, q_len, a_s, a_e, sample=100):
        #unused for now
        f1 = 0.0
        em = 0.0
        for sample in samples: # sample of size 100
            answers_dic = generate_answers(sess, model, sample_dataset, rev_vocab)
            result = evaluate(sample_dataset_dic, answers_dic)
            f1 += result['f1']
            em += result['exact_match']

        return f1, em

    def create_feed_dict(self, p,q, p_len, q_len, a_s=None, a_e=None, dropout=1.0):
        feed_dict = {
            self.dropout: dropout,
            self.p: p,
            self.q: q,
            self.p_len: p_len,
            self.q_len: q_len,
        }
        if a_s is not None:
            feed_dict[self.a_s] = a_s
        if a_e is not None:
            feed_dict[self.a_e] = a_e
        return feed_dict

    def predict_batch(self, sess, p, q, p_len, q_len):
        feed = self.create_feed_dict(p, q, p_len, q_len)
        a_s, a_e = sess.run([self.yp_start, self.yp_end], feed_dict=feed)
        return a_s, a_e

    def eval_batch(self, sess, p, q, p_len, q_len, a_s, a_e):
        feed = self.create_feed_dict( p, q, p_len, q_len, a_s=a_s,a_e=a_e, dropout=1.0)
        loss, norm, ys, ye = sess.run([self.loss, self.norm,self.yp_start, self.yp_end], feed_dict=feed)
        return loss, norm, ys, ye

    def train_batch(self, sess, p, q, p_len, q_len, a_s, a_e):
        feed = self.create_feed_dict( p, q, p_len, q_len, a_s=a_s,a_e=a_e, dropout=(1.0-FLAGS.dropout))
        _, loss, norm = sess.run([self.train_op, self.loss, self.norm], feed_dict=feed)
        return loss, norm

    def get_eval(self,sess, dataset, batch_size, sample=True):
        ''' if sample, take first batch only '''
        f1 = em = total = 0
        for i, batch in enumerate(get_minibatches(dataset, batch_size, shuffle=True)):
            p, q, p_len, q_len, a_s, a_e, p_raw = zip(*batch)
            loss, norm, ys, ye = self.eval_batch(sess, p, q, p_len, q_len, a_s, a_e)
            a_s_pred = np.argmax(ys, axis=1)
            a_e_pred = np.argmax(ye, axis=1)
            for i in range(len(batch)):
                #predicted a_s and a_e
                s_pred = a_s_pred[i]
                e_pred = a_e_pred[i]

                #ground truth lables
                a_raw = ' '.join(p_raw[i][a_s[i]:a_e[i]+1])
                pred_raw = ' '.join(p_raw[i][s_pred:e_pred+1])

                f1 += f1_score(pred_raw, a_raw)
                em += exact_match_score(pred_raw, a_raw)
                total += 1
            if sample:
                break

        em = 100.0 * em / total
        f1 = 100.0 * f1 / total
        return (f1, em, loss, norm)

    def run_epoch(self, sess, train_data, dev_data):
        for i, batch in enumerate(get_minibatches(train_data, FLAGS.batch_size)):
            p, q, p_len, q_len, a_s, a_e, _ = zip(*batch)
            with Timer("Train batch {}".format(i+1)):
                loss, norm = self.train_batch(sess, p, q, p_len, q_len, a_s, a_e)
                #logging.info("Train loss: {}, norm: {}".format(loss, norm))

            if i % FLAGS.sample_every == 0:
                f1, exact_match, loss, norm = self.get_eval(sess, train_data, FLAGS.sample_size, True)
                logging.info("[Batch {}] Train set sample_loss: {}, F1: {}, EM: {}, norm: {}".format(i+1, loss, f1, exact_match, norm))

                f1, exact_match, loss, norm = self.get_eval(sess, dev_data, FLAGS.sample_size, False)
                logging.info("[Batch {}] Val set loss: {}, F1: {}, EM: {}, norm: {}".format(i+1, loss, f1, exact_match, norm))
        print()

    def train(self, session, train_data, dev_data):
        best_score = 0.0
        for epoch in range(FLAGS.epochs):
            self.epoch = epoch + 1
            with Timer("training epoch {}/{}".format(epoch + 1, FLAGS.epochs)):
                logging.info("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
                self.run_epoch(session, train_data, dev_data)

                f1, exact_match, loss, _ = self.get_eval( session, dev_data, FLAGS.batch_size, False)
                logging.info("[Epoch {}] Val set loss: {}, F1: {}, EM: {}".format(epoch + 1, loss, f1, exact_match))

            if self.saver:
                epoch_output = pjoin(FLAGS.train_dir, 'epoch{}'.format(epoch+1), "model.weights")
                if not os.path.exists(epoch_output):
                    os.makedirs(epoch_output)
                logging.info("Saving model epoch %d in %s", epoch+1, epoch_output)
                self.saver.save(session, epoch_output)
        return best_score



