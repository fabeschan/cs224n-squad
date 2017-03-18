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
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
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
        self.model_output = FLAGS.train_dir + "/model.weights"
        self.initializer = tf.contrib.layers.xavier_initializer()

        with Timer("setup graph"):
            with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
                self.setup_placeholder()
                self.setup_system()
                self.setup_loss()
                self.setup_optimizer()
    ## attention layer from paper coattention
    def setup_attention_layer(self, D, Q):
        #add_column_Q = tf.zeros([None, FLAGS.question_size, 1])
        #print ("column_Q:", add_column_Q.get_shape())
        #add_column_D = tf.zeros([FLAGS.paragraph_size, 1])
        Q = tf.transpose(Q, perm=[0, 2, 1])
        # add tanh to Q
        #Q = tf.nn.tanh(Q)
        # softmax layer
        D = tf.transpose(D, perm=[0, 2, 1])
        # i.e. use dot-product scoring
        print("Q shape", Q.get_shape())
        L = tf.matmul(tf.transpose(D, perm=[0, 2, 1]), Q)  # much more complexity needed here (for example softmax scaling etc.)
        #s_max = tf.reduce_max(s, axis = 1, kddeep_dims=True)
        #s_min= tf.reduce_min(s, axis = 1, keep_dims=True)
        #s_mean = tf.reduce_mean(s, axis = 1, keep_dims=True)
        #s_enrich = tf.concat([s_max, s_min, s_mean], 1)

        print("L shape:",L.get_shape())
        A_Q = tf.nn.softmax(L, dim=1) # should be column-wise as sum(alpha_i) per paragraph-word is 1
        print("A_Q shape:", A_Q.get_shape())
        A_D = tf.nn.softmax(tf.transpose(L, perm=[0,2,1]), dim=1) # should be column-wise as sum(alpha_i) per question-word is 1

        C_Q = tf.matmul(D, A_Q)
        print("C_Q shape", C_Q.get_shape())

        '''
        # Add filter layer
        filterLayer = False
        if filterLayer:
            normed_p = tf.nn.l2_normalize(pp, dim=2) # normalize the 400 paragraph vectors
            normed_q = tf.nn.l2_normalize(qq, dim=2)
            cossim = tf.matmul(normed_p, tf.transpose(normed_q, [0, 2, 1]))
            rel = tf.reduce_max(cossim, axis = 2, keep_dims = True)
            p_emb_p = tf.multiply(rel, self.p_emb)
        else:
            p_emb_p = pp
        '''
        p_emb_p = D

        C_D = tf.matmul(tf.concat([Q, C_Q], 1), A_D)
        print("C_D shape:",C_D.get_shape())
        p_concat = tf.concat([p_emb_p, C_D], 1)
        return p_concat #tf.concat([s, s_enrich, tf.transpose(p_emb_p, perm = [0, 2, 1]) ], 1) #c, tf.transpose(pp, perm = [0, 2, 1]),

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
        #print([v.name for v in variables])
        gradients = optimizer.compute_gradients(self.loss, variables)
        gradients = [tup[0] for tup in gradients]
        if FLAGS.grad_clip:
            gradients, norms = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.norm = tf.global_norm(gradients)
        grads_and_vars = zip(gradients, variables)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def setup_system(self):
        ''' MPCM + COATT '''
        l_dim = FLAGS.perspective_units

        #define CELL
        self.cell = tf.contrib.rnn.BasicLSTMCell

        # add embeddings for question and paragraph
        # could tune this to be not trainable to boost speed
        self.embedding_mat = tf.Variable(self.pretrained_embeddings, name="embed", dtype=tf.float32)

        self.q_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.q), dtype=tf.float32)  # perhaps B-by-Q-by-d
        self.p_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.p), dtype=tf.float32)  # perhaps B-by-P-by-d

        cell_p_fwd = self.cell(FLAGS.hidden_size, state_is_tuple=True)
        cell_q_fwd = self.cell(FLAGS.hidden_size, state_is_tuple=True)
        cell_p_bwd = self.cell(FLAGS.hidden_size, state_is_tuple=True)
        cell_q_bwd = self.cell(FLAGS.hidden_size, state_is_tuple=True)
        cell_p_fwd = tf.contrib.rnn.DropoutWrapper(cell=cell_p_fwd, output_keep_prob=self.dropout)
        cell_q_fwd = tf.contrib.rnn.DropoutWrapper(cell=cell_q_fwd, output_keep_prob=self.dropout)
        cell_p_bwd = tf.contrib.rnn.DropoutWrapper(cell=cell_p_bwd, output_keep_prob=self.dropout)
        cell_q_bwd = tf.contrib.rnn.DropoutWrapper(cell=cell_q_bwd, output_keep_prob=self.dropout)

        # get bilstm encodings
        cur_batch_size = tf.shape(self.p)[0]


        print(("type1", (self.p_emb).get_shape()))
        # build the hidden representation for the question (fwd and bwd and stack them together)
        with tf.variable_scope("encode_q"):
            (output_q_fw, output_q_bw), (output_state_fw_q, output_state_bw_q) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_q_fwd,
                cell_bw=cell_q_bwd,
                inputs=self.q_emb,
                sequence_length=self.q_len,
                dtype=tf.float32

            )
            #print (output_q_fw.get_shape())
            #33print ("eeeeeeeeeeeeeeeeee", self.q_emb.get_shape())
            self.qq = tf.concat([output_q_fw, output_q_bw], 2)  # 2h_dim dimensional representation over each word in question
            # self.qq = tf.reshape(output_state_fw_q, shape = [self.batch_size, 1, 2*h_dim]) + tf.reshape(output_state_bw_q, shape = [self.batch_size, 1, 2*h_dim])  # try only using "end representation"  as question summary vector
            #print(("type11", (self.qq).get_shape()))
        with tf.variable_scope("encode_p"):
            outputs_p, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_p_fwd,
                cell_bw=cell_p_bwd,
                initial_state_fw=output_state_fw_q,
                inputs=self.p_emb,
                sequence_length=self.p_len,
                dtype=tf.float32
            )
            self.pp = tf.concat(outputs_p, 2)  # 2h_dim dimensional representation over each word in context-paragraph

        # need to mix qq and pp to get an attention matrix (question-words)-by-(paragraph-words) dimensional heat-map like matrix for each example
        # this attention matrix will allow us to localize the interesting parts of the paragraph (they will peak) and help us identify the patch corresponding to answer
        # visually the patch will ideally start at start-index and decay near end-index
        self.coatt_layer = self.setup_attention_layer(self.pp, self.qq)

        #  predictions obtain by applying softmax over something (attention vals - should be something like dim(question-words)-by-dim(paragraph)
        # currently a B-by-QMAXLEN-by-PMAXLEN tensor


        # apply another LSTM layer before softmx
        cell_final = self.cell(FLAGS.hidden_size*4, state_is_tuple=True)
        U, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_final,
            cell_bw=cell_final,
            inputs=tf.transpose(self.coatt_layer, perm = [0, 2, 1]),
            sequence_length=self.p_len ,
            dtype=tf.float32
        )
        #print ("ddddddddddddddddddddd", out_lstm.get_shape())
        #print(("dim_att", (self.coatt_layer).get_shape()))
        U = tf.concat(U, 2)
        #lstm_final = tf.transpose(, perm = [0, 2, 1])  # 2*2h_dim dimensional representation over each word in context-paragraph
        print("U", U.get_shape())
        #lstm_final = tf.transpose(self.att, perm=[0, 2, 1])
        U = tf.nn.dropout(U, self.dropout)
        U_reshape = tf.reshape(U, shape=[-1,FLAGS.hidden_size*8])

        # softmax layer
        W_start = tf.get_variable("W_start", shape=[8*FLAGS.hidden_size, 1], initializer=self.initializer)
        b_start = tf.Variable(tf.zeros([FLAGS.paragraph_size]))
        self.logits_start_1 = tf.reshape(tf.matmul(U_reshape, W_start), shape=[-1, FLAGS.paragraph_size]) + b_start
        print ("logits", self.logits_start_1)
        self.logits_start = tf.nn.softmax(self.logits_start_1, -1)
        self.yp_start = self.logits_start
        #self.U_s_index = tf.argmax(self.logits_start,1)

        W_end = tf.get_variable("W_end", shape=[8*FLAGS.hidden_size, 1], initializer=self.initializer)
        b_end = tf.Variable(tf.zeros([FLAGS.paragraph_size]))
        self.logits_end_1 = (tf.reshape(tf.matmul(U_reshape, W_end), shape=[-1, FLAGS.paragraph_size]) + b_end)
        self.logits_end = tf.nn.softmax(self.logits_end_1, -1)
        #self.U_e_index = tf.argmax(self.logits_end,1)
        self.yp_end = self.logits_end

        '''
        U_s_e = tf.gather_nd(U, indices = [[U_s_index, U_e_index]])
        print ("U_s_e",U_s_e.get_shape())

        cell_decode = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size*2)

        with tf.variable_scope("decode"):
            h_1, _ = tf.nn.dynamic_rnn(
                cell=cell_decode,
                inputs=U_s_e,
                sequence_length=self.p_len,
                dtype=tf.float32
            )

        r_input_1 = tf.concat([h_1, U_s_e], 2)
        W_D = tf.get_variable("W_D", shape=[FLAGS.hidden_size*10, FLAGS.hidden_size*2], initializer=self.initializer)
        r_input = tf.reshape(r_input, shape=[-1, FLAGS.hidden_size*10])
        print("r_input", r_input.get_shape())
        r = tf.nn.tanh(tf.matmul(r_input, W_D))
        r = tf.tile(r, multiples=tf.constant(FLAGS.paragraph_size))
        print ("r", r.get_shape())

        m1_input = tf.concat([U, r], )
        print("r ", r.get_shape())
        dim_final_layer = int(lstm_final.get_shape()[2])
        final_layer_ = tf.reshape(lstm_final, shape=[-1, dim_final_layer])

        # softmax layer
        W_start = tf.get_variable("W_start", shape=[dim_final_layer, 1], initializer=self.initializer)
        b_start = tf.Variable(tf.zeros([FLAGS.paragraph_size]))
        self.logits_start_1 = tf.reshape(tf.matmul(final_layer_, W_start), shape=[cur_batch_size, FLAGS.paragraph_size]) + b_start
        self.yp_start_1 = tf.nn.softmax(self.logits_start_1)

        W_end = tf.get_variable("W_end", shape=[dim_final_layer, 1], initializer=self.initializer)
        b_end = tf.Variable(tf.zeros([FLAGS.paragraph_size]))
        self.logits_end_1 = (tf.reshape(tf.matmul(final_layer_, W_end), shape=[cur_batch_size, FLAGS.paragraph_size]) + b_end)
        self.yp_end_1 = tf.nn.softmax(self.logits_end_1)
        '''
        # MPCM

        '''
        # Add filter layer
        with tf.variable_scope("encode_filter"):
            filterLayer = True
            if filterLayer:
                normed_p = tf.nn.l2_normalize(self.p_emb, dim=2) # normalize the 400 paragraph vectors
                normed_q = tf.nn.l2_normalize(self.q_emb, dim=2)
                cossim = tf.matmul(normed_p, tf.transpose(normed_q, [0, 2, 1]))
                rel = tf.reduce_max(cossim, axis = 2, keep_dims = True)
                self.p_emb_p = tf.multiply(rel, self.p_emb)
            else:
                self.p_emb_p = self.p_emb

        # add dropout after filter layer
        self.q_emb = tf.nn.dropout(self.q_emb, self.dropout)
        self.p_emb_p = tf.nn.dropout(self.p_emb_p, self.dropout)

        cell = self.cell(FLAGS.hidden_size, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.dropout)

        # Context Representation Layer
        with tf.variable_scope("encode_p_mpcm"):
            (output_p_fw, output_p_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell,
                inputs=self.p_emb_p,
                initial_state_fw=output_state_fw_q,
                sequence_length=self.p_len,
                dtype=tf.float32
            )

            # Multi-perspective context matching layer
            W1 = tf.get_variable(
                "W1",
                shape=[1, 1, FLAGS.hidden_size, l_dim],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            W2 = tf.get_variable(
                "W2",
                shape=[1, 1, FLAGS.hidden_size, l_dim],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            output_p_fw = tf.expand_dims(output_p_fw, 3)
            tp1 = tf.nn.l2_normalize(tf.multiply(output_p_fw, W1), dim=2)
            qs1 = output_q_fw[:, FLAGS.question_size - 1, :]
            qs1 = tf.expand_dims(qs1, 1)
            qs1 = tf.expand_dims(qs1, 3)
            tq1 = tf.nn.l2_normalize(tf.multiply(qs1, W1), dim=2)
            m1 = tf.multiply(tp1, tq1)
            m1_full = tf.reduce_sum(m1, axis=2)
            m1_max = tf.reduce_max(m1, axis=2)
            m1_mean = tf.reduce_mean(m1, axis=2)


            output_p_bw = tf.expand_dims(output_p_bw, 3)
            tp2 = tf.nn.l2_normalize(tf.multiply(output_p_bw, W2), dim=2)
            qs2 = output_q_bw[:, 0, :]
            qs2 = tf.expand_dims(qs2, 1)
            qs2 = tf.expand_dims(qs2, 3)
            tq2 = tf.nn.l2_normalize(tf.multiply(qs2, W2), dim=2)
            m2 = tf.multiply(tp2, tq2)
            m2_full = tf.reduce_sum(m2, axis=2)
            m2_max = tf.reduce_max(m2, axis=2)
            m2_mean = tf.reduce_mean(m2, axis=2)

            m = tf.concat([m1_full, m1_max, m1_mean, m2_full, m2_max, m2_mean], axis=2)
            m = tf.nn.dropout(m, self.dropout)

            # Aggregation layer
            cur_batch_size = tf.shape(self.p)[0];

            with tf.variable_scope("mix"):
                outputs_mix, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell,
                    cell_bw=cell,
                    inputs=m,
                    sequence_length= self.p_len,
                    dtype=tf.float32)
                final_layer = tf.concat(outputs_mix, 2)

            dim_final_layer = int(final_layer.get_shape()[2])
            final_layer_ = tf.reshape(final_layer, shape=[-1, dim_final_layer])

            # Prediction layer
            # this should be replaced by a NN as in the paper later
            W_start_ = tf.get_variable("W_start_", shape=[dim_final_layer, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_start_ = tf.Variable(tf.zeros([FLAGS.paragraph_size]))
            mixed_start = tf.matmul(final_layer_, W_start_)
            mixed_start = tf.reshape(mixed_start, shape=[-1, FLAGS.paragraph_size])
            self.logits_start_2 = mixed_start + b_start_
            self.yp_start_2 = tf.nn.softmax(self.logits_start_2)

            W_end_ = tf.get_variable("W_end_", shape=[dim_final_layer, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_end_ = tf.Variable(tf.zeros([FLAGS.paragraph_size]))
            mixed_end = tf.matmul(final_layer_, W_end_)
            mixed_end = tf.reshape(mixed_end, shape=[-1, FLAGS.paragraph_size])

        self.logits_end_2 = mixed_end + b_end_
        self.yp_end_2 = tf.nn.softmax(self.logits_end_2)
        # add logits (before softmaxO)
        self.logits_end = self.logits_end_1 + self.logits_end_2
        self.logits_start = self.logits_start_1 + self.logits_start_2
        # multiply probabilities
        self.yp_start = tf.multiply(self.yp_start_1, self.yp_start_2)
        self.yp_end = tf.multiply(self.yp_end_1, self.yp_end_2)
        '''
    def setup_loss(self):
        #may need to do some reshaping here
        # Someone replaced yp_start with logits_start in loss. Don't really follow the change. Setting it back to original.
        with vs.variable_scope("loss"):
            self.loss_start_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start_1, labels=self.a_s))
            self.loss_end_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end_1, labels=self.a_e))
            #self.loss_start_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start_2, labels=self.a_s))
            #self.loss_end_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end_2, labels=self.a_e))
            # compute span l2 loss
            a_s_p = tf.argmax(self.yp_start, axis=1)
            a_e_p = tf.argmax(self.yp_end, axis=1)
            self.loss_span = tf.reduce_mean(tf.nn.l2_loss(tf.cast(self.a_e - self.a_s + 1, tf.float32) - tf.cast(a_s_p - a_e_p, tf.float32)))
            #self.loss = tf.add(self.loss_start_1, self.loss_end_1) + tf.add(self.loss_start_2, self.loss_end_2) + FLAGS.l2_lambda * self.loss_span
            self.loss = tf.add(self.loss_start_1, self.loss_end_1) + FLAGS.l2_lambda * self.loss_span

    def evaluate_answer(self, session,  p, q, p_len, q_len, a_s, a_e, sample=100):
        f1 = 0.0
        em = 0.0

        #unused for now
        for sample in samples: # sample of size 100
            answers_dic = generate_answers(sess, model, sample_dataset, rev_vocab) # returns dictionary to be fed in evaluate
            result = evaluate(sample_dataset_dic, answers_dic) # takes dictionaries of form nswers[uuid] = "real answer"
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
        (s_index, e_start) = sess.run([self.yp_start, self.yp_end], feed_dict=feed)
        return (s_index, e_start)

    def train_batch(self, sess, p, q, p_len, q_len, a_s, a_e):
        feed = self.create_feed_dict( p, q, p_len, q_len, a_s=a_s,a_e=a_e, dropout=(1.0-FLAGS.dropout))
        _, loss, norm = sess.run([self.train_op, self.loss, self.norm], feed_dict=feed)
        return loss, norm

    def run_epoch(self, sess, train_examples, dev_set):
        for i, batch in enumerate(get_minibatches(train_examples, FLAGS.batch_size)):
            p, q, p_len, q_len, a_s, a_e, _ = zip(*batch)
            #if len(batch) != FLAGS.batch_size:
            #    continue
            with Timer("train batch {}".format(i)):
                loss, norm = self.train_batch(sess, p, q, p_len, q_len, a_s, a_e)
                logging.info("train loss: {}, norm: {}".format(loss, norm))
        print("")

        logging.info("Evaluating on development data")
        f1 = exact_match = total = 0
        for i, batch in enumerate(get_minibatches(dev_set, FLAGS.batch_size)):
            # Only use P and Q
            p, q, p_len, q_len, a_s, a_e, p_raw = zip(*batch)
            (ys, ye) = self.predict_batch(sess, p, q, p_len, q_len)
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
                exact_match += exact_match_score(pred_raw, a_raw)
                total += 1

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        logging.info("Entity level F1/EM: %.2f/%.2f", f1, exact_match)

        return f1

    def train(self, session, train_data, dev_data):
        saver = tf.train.Saver()
        best_score = 0.0
        for epoch in range(FLAGS.epochs):
            with Timer("training epoch {}/{}".format(epoch + 1, FLAGS.epochs)):
                logging.info("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
                score = self.run_epoch(session, train_data, dev_data)
                if score > best_score:
                    best_score = score
                    if saver:
                        logging.info("New best score! Saving model in %s", self.model_output)
                        saver.save(session, self.model_output)
                print("")
            logging.info("Best f1 score detected this run : %s ", best_score)
        return best_score



