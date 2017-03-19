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
from ops import highway_maxout, batch_linear

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
        self.cell = tf.contrib.rnn.BasicLSTMCell
        self.initializer = tf.contrib.layers.xavier_initializer()

        with Timer("setup graph"):
            with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
                self.setup_placeholder()
                self.encoder()
                self.decoder()
                self.setup_loss()
                self.setup_optimizer()
        self.saver = tf.train.Saver()

    def encoder(self):

        # add embeddings for question and paragraph
        # could tune this to be not trainable to boost speed
        self.embedding_mat = tf.Variable(self.pretrained_embeddings, name="embed", dtype=tf.float32)
        fn_embedding = lambda x: tf.nn.embedding_lookup(self.embedding_mat, x)
        self.p_emb = tf.map_fn(lambda x: fn_embedding(x), self.p, dtype=tf.float32) # (None, paragraph_size, emb_size)    
        self.q_emb = tf.map_fn(lambda x: fn_embedding(x), self.q, dtype=tf.float32) # (None, question_size, emb_size)         



        # LSTM for question, get self.Q
        with tf.variable_scope("encode_q"):
            cell_q = self.cell(FLAGS.hidden_size, state_is_tuple=True)
            cell_q = tf.contrib.rnn.DropoutWrapper(cell=cell_q, output_keep_prob=self.dropout)
            Q, output_state_q = tf.nn.dynamic_rnn(
                cell=cell_q,
                inputs=self.q_emb,
                sequence_length=self.q_len,
                dtype=tf.float32
            )
            # Q: (None, question_size, hidden_size)
            # deal D: add sentinel and do tanh; D: (None, hidden_size, question_size+1)
            fn_add_column = lambda x: tf.concat([x, tf.zeros([1, FLAGS.hidden_size], dtype=tf.float32)], 0)
            #Q = tf.map_fn(lambda x: fn_add_column(x), Q, dtype=tf.float32)
            #Q = tf.tanh(batch_linear(Q, FLAGS.question_size+1, True))
            Q = tf.transpose(Q, perm=[0, 2, 1])


        # LSTM for question, get self.D
        with tf.variable_scope("encode_p"):
            cell_p = self.cell(FLAGS.hidden_size, state_is_tuple=True)
            cell_p = tf.contrib.rnn.DropoutWrapper(cell=cell_p, output_keep_prob=self.dropout)
            D, output_state_p = tf.nn.dynamic_rnn(
                cell=cell_p,
                initial_state=output_state_q,
                inputs=self.p_emb,
                sequence_length=self.p_len,
                dtype=tf.float32
            )
            # D: (None, paragraph_size, hidden_size)
            # deal D: add sentinel; D: (None, hidden_size, paragraph_size+1)
            #fn_add_column = lambda x: tf.concat([x, tf.zeros([1, FLAGS.hidden_size], dtype=tf.float32)], 0)
            #D = tf.map_fn(lambda x: fn_add_column(x), D, dtype=tf.float32)
            D = tf.transpose(D, perm=[0, 2, 1])

        # Set up Coatt_layer 
        with tf.variable_scope('coattention'):
            L = tf.matmul(tf.transpose(D, perm=[0, 2, 1]), Q) # L shape: (?, paragraph_size+1, question_size+1)
            A_Q = tf.nn.softmax(L, dim=1) # L shape: (?, paragraph_size+1, question_size+1)
            A_D = tf.nn.softmax(tf.transpose(L, perm=[0,2,1]), dim=1)
            C_Q = tf.matmul(D, A_Q) # C_Q shape (?, hidden_size, question_size+1)
            C_D = tf.matmul(tf.concat([Q, C_Q], 1), A_D) # C_D shape: (?, hidden_size*2, paragraph_size+1)s
            coatt_layer = tf.concat([D, C_D], 1)
        print ("done the coatt_layer")

        # Bidirectional LSTM layer 
        with tf.variable_scope('encoder'):
            cell_final_fw = self.cell(FLAGS.hidden_size, state_is_tuple=True)
            cell_final_bw = self.cell(FLAGS.hidden_size, state_is_tuple=True)
            U, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_final_fw,
                cell_bw=cell_final_bw,
                inputs=tf.transpose(coatt_layer, perm = [0, 2, 1]),
                sequence_length=self.p_len,
                dtype=tf.float32
            )
            U = tf.concat(U, 2) 
            self.U = tf.nn.dropout(U, self.dropout)
            self.U = self.U[:,:FLAGS.paragraph_size,:]
            print("U", self.U.get_shape())


    

    def decoder(self):
        # def select(u, pos, idx):
        #     u_idx = tf.gather(u, idx)
        #     pos_idx = tf.gather(pos, idx)
        #     return tf.reshape(tf.gather(u_idx, pos_idx), [-1])
        # # selector layer:
        # with tf.variable_scope('selector'):
        #     cell_q = self.cell(FLAGS.hidden_size, state_is_tuple=True)
        #     highway_alpha = highway_maxout(FLAGS.hidden_size, FLAGS.p)
        #     highway_beta = highway_maxout(FLAGS.hidden_size, FLAGS.p)
        #     # reshape self._u, (context, batch_size, 2*hidden_size)
        #     U = tf.transpose(self.U[:,:FLAGS.paragraph_size,:], perm=[1, 0, 2])
        #     print("U", U.get_shape())
        #     loop_until = tf.to_int32(np.array(range(FLAGS.batch_size)))
        #     # initial guess for index position for each batch
        #     initial_guess = np.zeros((2, FLAGS.batch_size))
        #     s, e = tf.split(initial_guess, 2, 0)
        #     fn = lambda idx: select(self._u, s, idx)
        #     u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)
        #     fn = lambda idx: select(self._u, e, idx)
        #     u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)


        # softmax layer
        U_reshape = tf.reshape(self.U, shape=[-1,FLAGS.hidden_size*2])
        W_start = tf.get_variable("W_start", shape=[2*FLAGS.hidden_size, 1], initializer=self.initializer)
        b_start = tf.Variable(tf.zeros([FLAGS.paragraph_size]))
        self.logits_start_1 = tf.reshape(tf.matmul(U_reshape, W_start), shape=[-1, FLAGS.paragraph_size]) + b_start
        print ("logits", self.logits_start_1)
        self.logits_start = tf.nn.softmax(self.logits_start_1, -1)
        self.yp_start = self.logits_start_1
        print ("logis_start", self.logits_start_1.get_shape())
        #self.U_s_index = tf.argmax(self.logits_start,1)

        W_end = tf.get_variable("W_end", shape=[2*FLAGS.hidden_size, 1], initializer=self.initializer)
        b_end = tf.Variable(tf.zeros([FLAGS.paragraph_size]))
        self.logits_end_1 = (tf.reshape(tf.matmul(U_reshape, W_end), shape=[-1, FLAGS.paragraph_size]) + b_end)
        self.logits_end = tf.nn.softmax(self.logits_end_1, -1)
        #self.U_e_index = tf.argmax(self.logits_end,1)
        print ("logis_end", self.logits_end_1.get_shape())
        self.yp_end = self.logits_end_1
    def setup_placeholder(self):
        self.dropout = tf.placeholder(tf.float32, shape=())
        self.q = tf.placeholder(tf.int32, [None, FLAGS.question_size], name="question")
        self.p = tf.placeholder(tf.int32, [None, FLAGS.paragraph_size], name="paragraph")
        self.a_s = tf.placeholder(tf.int32, [None], name="answer_start")
        self.a_e = tf.placeholder(tf.int32, [None], name="answer_end")
        self.p_len = tf.placeholder(tf.int32, [None], name="paragraph_len")
        self.q_len = tf.placeholder(tf.int32, [None], name = "question_len")
        #self.add_column_question = tf.placeholder(tf.int32, [None, FLAGS.question_size], name="add_column_question")
        #self.add_column_paragraph = tf.placeholder(tf.int32, [None, FLAGS.paragraph_size], name="add_column_paragraph")

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
    def setup_loss(self):
        with vs.variable_scope("loss"):
            print ("self.a_s", self.a_s.get_shape())
            print (self.logits_start_1.get_shape())
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



