from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import QASystem
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data
from qa_data import PAD_ID

import logging
from utils import pad_sequences

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.0, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 300, "Batch size to use during eval.")
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
tf.app.flags.DEFINE_integer("paragraph_size", 400, "Max Context Paragraph Length")
tf.app.flags.DEFINE_integer("hidden_size", 200, "size of hidden layer h_i")
tf.app.flags.DEFINE_integer("sample_every", 100, "every 100 batch to plot one result")
tf.app.flags.DEFINE_integer("sample_size", 100, "size of sample")

tf.app.flags.DEFINE_integer("perspective_units", 50, "Number of lstm representation h_i")
tf.app.flags.DEFINE_bool("grad_clip", True, "whether or not to clip the gradients")
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "lambda constant for regularization")

tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
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

def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []
    context_token_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)
                context_token_data.append(context_tokens)

    return context_data, query_data, question_uuid_data, context_token_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data, context_tokens = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data, context_tokens

def get_minibatches(data, batch_size=-1):
    batch = []
    indices = range(len(data))
    for i in indices:
        batch.append(data[i])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch):
        yield batch


def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    logging.info("Starting answering")
    answers = {}
    zipped = zip(*dataset)
    num_batches = (len(zipped) + FLAGS.batch_size - 1) // FLAGS.batch_size
    for k, batch in enumerate(get_minibatches(zipped, FLAGS.batch_size)):
        context_data, question_data, question_uuid_data, context_tokens = zip(*batch)
        p, q = [], []
        p_len, q_len = [], []
        for i in range(len(batch)):
            q.append(question_data[i].split())
            q_len.append(min(FLAGS.question_size, len(question_data[i].split())))

            p.append(context_data[i].split())
            p_len.append(min(FLAGS.paragraph_size, len(context_data[i].split())))

        q = pad_sequences(q, maxlen=FLAGS.question_size, value=PAD_ID, padding="post")
        p = pad_sequences(p, maxlen=FLAGS.paragraph_size, value=PAD_ID, padding="post")

        ys, ye = model.predict_batch(sess, p, q, p_len, q_len)

        a_s_pred = np.argmax(ys, axis=1)
        a_e_pred = np.argmax(ye, axis=1)
        for i in range(len(batch)):
            #predicted a_s and a_e
            s_pred = a_s_pred[i]
            e_pred = a_e_pred[i]

            uuid = question_uuid_data[i]
            pred_raw = ' '.join(context_tokens[i][s_pred:e_pred+1])
            answers[uuid] = pred_raw
        logging.info("Finished answering batch {} of {}".format(k+1, num_batches))
    return answers



def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def main(_):

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, question_data, question_uuid_data, context_tokens = prepare_dev(dev_dirname, dev_filename, vocab)
    dataset = (context_data, question_data, question_uuid_data, context_tokens)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    embeddings = np.load(FLAGS.data_dir + '/glove.trimmed.' + str(FLAGS.embedding_size) + '.npz')
    pretrained_embeddings = embeddings['glove']

    qa = QASystem(FLAGS, pretrained_embeddings, vocab_dim=len(vocab.keys()))

    with tf.Session() as sess:
        train_dir = get_normalized_train_dir(FLAGS.train_dir)
        initialize_model(sess, qa, train_dir)
        answers = generate_answers(sess, qa, dataset, rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
