# coding=utf-8
from __future__ import print_function

import argparse
import os
import re
import sys
import time

import tensorflow as tf
import namespace_utils
from SentenceMatchDataStream import SentenceMatchDataStream
from SentenceMatchModelGraph import SentenceMatchModelGraph
from vocab_utils import Vocab

FLAGS = None


def collect_vocabs(train_path):
    all_labels = set()
    all_words = set()
    print("Du Qu")
    for line in open(train_path, 'r'):
        line = line.strip()
        if line.startswith('-'): continue
        items = re.split("\t", line)
        if len(items) != 3:
            print(1)
            continue
        label = items[0]
        sentence1 = re.split("\\s+", items[1].lower())  # 分英文单词
        sentence2 = re.split("\\s+", items[2].lower())
        all_labels.add(label)
        all_words.update(sentence1)
        all_words.update(sentence2)
    print("complete")
    return (all_words, all_labels)


def evaluate(dataStream, valid_graph, sess, outpath=None):
    if outpath is not None: outfile = open(outpath, 'wt')
    total_tags = 0.0
    correct_tags = 0.0
    dataStream.reset()
    for batch_index in iter(range(dataStream.get_num_batch())):
        cur_dev_batch = dataStream.get_batch(batch_index)

        (label_id_batch, word_idx_1_batch, word_idx_2_batch, sent1_length_batch, sent2_length_batch) = cur_dev_batch

        feed_dict = {
            valid_graph.get_truth(): label_id_batch,
            valid_graph.get_question_lengths(): sent1_length_batch,
            valid_graph.get_passage_lengths(): sent2_length_batch,
            valid_graph.get_in_question_words(): word_idx_1_batch,
            valid_graph.get_in_passage_words(): word_idx_2_batch,
        }

        total_tags += len(label_id_batch)
        correct_tags += sess.run(valid_graph.get_eval_correct(), feed_dict=feed_dict)

    if outpath is not None: outfile.close()

    accuracy = correct_tags / total_tags * 100
    return accuracy


def output_probs(probs, label_vocab):
    label_pos = None
    prob_pos = -1
    for i in iter(range(probs.size)):
        if probs[i] > prob_pos:
            prob_pos = probs[i]  # 0.99
            label_pos = label_vocab.getWord(i)  # 0
    if label_pos == '0':
        prob_pos = 1 - prob_pos
        label_pos = 1 - int(label_pos)
    out_string = str(label_pos) + ":" + str(prob_pos)
    return out_string


def main(_):
    print('Configurations:')
    print(FLAGS)  # 打印各个参数

    root_path = FLAGS.root_path
    train_path = root_path + FLAGS.train_path
    dev_path = root_path + FLAGS.dev_path
    test_path = root_path + FLAGS.test_path
    word_vec_path = root_path + FLAGS.word_vec_path
    model_dir = root_path + FLAGS.model_dir

    if tf.gfile.Exists(model_dir + '/mnist_with_summaries'):
        print("delete summaries")
        tf.gfile.DeleteRecursively(model_dir + '/mnist_with_summaries')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    path_prefix = model_dir + "/SentenceMatch.{}".format(FLAGS.suffix)

    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")  # 保存参数

    best_path = path_prefix + '.best.model'
    label_path = path_prefix + ".label_vocab"
    has_pre_trained_model = False
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("-------has_pre_trained_model--------")
        print(ckpt.model_checkpoint_path)
        has_pre_trained_model = True

    ############# build vocabs#################
    print('Collect words, chars and labels ...')
    (all_words, all_labels) = collect_vocabs(train_path)
    print('Number of words: {}'.format(len(all_words)))
    print('Number of labels: {}'.format(len(all_labels)))

    word_vocab = Vocab(pattern='word')  # 定义一个类
    word_vocab.patternWord(word_vec_path, model_dir)
    label_vocab = Vocab(pattern="label")
    label_vocab.patternLabel(all_labels, label_path)

    print('word_vocab shape is {}'.format(word_vocab.word_vecs.shape))
    print('tag_vocab shape is {}'.format(label_vocab.word_vecs.shape))
    num_classes = len(all_labels)

    if FLAGS.wo_char: char_vocab = None
    #####  Build SentenceMatchDataStream  ################
    print('Build SentenceMatchDataStream ... ')
    trainDataStream = SentenceMatchDataStream(train_path, word_vocab=word_vocab, label_vocab=label_vocab,
                                              batch_size=FLAGS.batch_size, isShuffle=True, isLoop=True, isSort=False,
                                              max_sent_length=FLAGS.max_sent_length)

    devDataStream = SentenceMatchDataStream(dev_path, word_vocab=word_vocab, label_vocab=label_vocab,
                                            batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=False,
                                            max_sent_length=FLAGS.max_sent_length)

    testDataStream = SentenceMatchDataStream(test_path, word_vocab=word_vocab, label_vocab=label_vocab,
                                             batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=False,
                                             max_sent_length=FLAGS.max_sent_length)

    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))

    sys.stdout.flush()

    best_accuracy = 0.0
    init_scale = 0.01
    g_2 = tf.Graph()
    with g_2.as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_graph = SentenceMatchModelGraph(
                num_classes,
                word_vocab=word_vocab,
                dropout_rate=FLAGS.dropout_rate,
                learning_rate=FLAGS.learning_rate,
                optimize_type=FLAGS.optimize_type,
                lambda_l2=FLAGS.lambda_l2,
                with_word=True,
                context_lstm_dim=FLAGS.context_lstm_dim,
                aggregation_lstm_dim=FLAGS.aggregation_lstm_dim,
                is_training=True,
                MP_dim=FLAGS.MP_dim,
                context_layer_num=FLAGS.context_layer_num,
                aggregation_layer_num=FLAGS.aggregation_layer_num,
                fix_word_vec=FLAGS.fix_word_vec,
                with_filter_layer=FLAGS.with_filter_layer,
                with_highway=FLAGS.with_highway,
                with_match_highway=FLAGS.with_match_highway,
                with_aggregation_highway=FLAGS.with_aggregation_highway,
                highway_layer_num=FLAGS.highway_layer_num,
                with_lex_decomposition=FLAGS.with_lex_decomposition,
                lex_decompsition_dim=FLAGS.lex_decompsition_dim,
                with_left_match=(not FLAGS.wo_left_match),
                with_right_match=(not FLAGS.wo_right_match),
                with_full_match=(not FLAGS.wo_full_match),
                with_maxpool_match=(not FLAGS.wo_maxpool_match),
                with_attentive_match=(not FLAGS.wo_attentive_match),
                with_max_attentive_match=(not FLAGS.wo_max_attentive_match))
            tf.summary.scalar("Training Loss", train_graph.get_loss())

        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(num_classes,
                                                  word_vocab=word_vocab,
                                                  dropout_rate=FLAGS.dropout_rate,
                                                  learning_rate=FLAGS.learning_rate,
                                                  optimize_type=FLAGS.optimize_type,
                                                  lambda_l2=FLAGS.lambda_l2,
                                                  with_word=True,
                                                  context_lstm_dim=FLAGS.context_lstm_dim,
                                                  aggregation_lstm_dim=FLAGS.aggregation_lstm_dim,
                                                  is_training=False,
                                                  MP_dim=FLAGS.MP_dim,
                                                  context_layer_num=FLAGS.context_layer_num,
                                                  aggregation_layer_num=FLAGS.aggregation_layer_num,
                                                  fix_word_vec=FLAGS.fix_word_vec,
                                                  with_filter_layer=FLAGS.with_filter_layer,
                                                  with_highway=FLAGS.with_highway,
                                                  with_match_highway=FLAGS.with_match_highway,
                                                  with_aggregation_highway=FLAGS.with_aggregation_highway,
                                                  highway_layer_num=FLAGS.highway_layer_num,
                                                  with_lex_decomposition=FLAGS.with_lex_decomposition,
                                                  lex_decompsition_dim=FLAGS.lex_decompsition_dim,
                                                  with_left_match=(not FLAGS.wo_left_match),
                                                  with_right_match=(not FLAGS.wo_right_match),
                                                  with_full_match=(not FLAGS.wo_full_match),
                                                  with_maxpool_match=(not FLAGS.wo_maxpool_match),
                                                  with_attentive_match=(not FLAGS.wo_attentive_match),
                                                  with_max_attentive_match=(not FLAGS.wo_max_attentive_match))

        initializer = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # vars_ = {}
        # for var in tf.global_variables():
        #     if "word_embedding" in var.name: continue
        #     vars_[var.name.split(":")[0]] = var
        # saver = tf.train.Saver(vars_)

        sess = tf.Session()

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(model_dir + '/mnist_with_summaries/train', sess.graph)
        sess.run(initializer)

        if has_pre_trained_model:
            print("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            print("DONE!")

        print('Start the training loop.')
        train_size = trainDataStream.get_num_batch()
        max_steps = train_size * FLAGS.max_epochs
        total_loss = 0.0
        start_time = time.time()
        for step in iter(range(max_steps)):
            cur_batch = trainDataStream.nextBatch()
            (label_id_batch, word_idx_1_batch, word_idx_2_batch, sent1_length_batch, sent2_length_batch) = cur_batch
            feed_dict = {
                train_graph.get_truth(): label_id_batch,
                train_graph.get_question_lengths(): sent1_length_batch,
                train_graph.get_passage_lengths(): sent2_length_batch,
                train_graph.get_in_question_words(): word_idx_1_batch,
                train_graph.get_in_passage_words(): word_idx_2_batch,
            }

            # in_question_repres,in_ques=sess.run([train_graph.in_question_repres,train_graph.in_ques],feed_dict=feed_dict)
            # print(in_question_repres,in_ques)
            # break

            _, loss_value, summary = sess.run([train_graph.get_train_op(), train_graph.get_loss(), merged],
                                              feed_dict=feed_dict)
            total_loss += loss_value

            if step % 5000 == 0:
                # train_writer.add_summary(summary, step)
                print("step:", step, "loss:", loss_value)

            if (step + 1) % trainDataStream.get_num_batch() == 0 or (step + 1) == max_steps:
                print()
                duration = time.time() - start_time
                start_time = time.time()
                print('Step %d: loss = %.2f (%.3f sec)' % (step, total_loss, duration))
                total_loss = 0.0

                print('Validation Data Eval:')
                accuracy = evaluate(devDataStream, valid_graph, sess)
                print("Current accuracy is %.2f" % accuracy)
                if accuracy >= best_accuracy:
                    print('Saving model since it\'s the best so far')
                    best_accuracy = accuracy
                    saver.save(sess, best_path)

    print("Best accuracy on dev set is %.2f" % best_accuracy)


if __name__ == '__main__':
    ## modify: batch_size,learning_rate,dropout_rate,highway_layer_num,
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default="/home/haojianyong/file_1/biMPM/src/",
                        help='Path to root.')
    parser.add_argument('--train_path', type=str, default="data/train.dat",
                        help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, default="data/dev.dat",
                        help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, default="data/test.dat",
                        help='Path to the test set.')
    parser.add_argument('--word_vec_path', type=str, default="data/wordvec.vec",
                        help='Path the to pre-trained word vector model.')
    parser.add_argument('--model_dir', type=str, default="models",
                        help='Need to the whole path, Directory to save model files.')
    parser.add_argument('--max_sent_length', type=int, default=25,
                        help='Maximum number of words within each sentence.')

    parser.add_argument('--batch_size', type=int, default=64, help='Number of instances in each batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout ratio.')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs for training.')
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--char_emb_dim', type=int, default=20, help='Number of dimension for character embeddings.')
    parser.add_argument('--char_lstm_dim', type=int, default=100,
                        help='Number of dimension for character-composed embeddings.')
    parser.add_argument('--context_lstm_dim', type=int, default=100,
                        help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=300,
                        help='Number of dimension for aggregation layer.')
    parser.add_argument('--MP_dim', type=int, default=10, help='Number of perspectives for matching vectors.')
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')
    parser.add_argument('--aggregation_layer_num', type=int, default=1,
                        help='Number of LSTM layers for aggregation layer.')
    parser.add_argument('--context_layer_num', type=int, default=1,
                        help='Number of LSTM layers for context representation layer.')
    parser.add_argument('--highway_layer_num', type=int, default=1, help='Number of highway layers.')
    parser.add_argument('--suffix', type=str, default='sample', help='Suffix of the model name.')
    parser.add_argument('--fix_word_vec', default=True, help='Fix pre-trained word embeddings during training.',
                        action='store_true')
    parser.add_argument('--with_highway', default=True, help='Utilize highway layers.', action='store_true')
    # with_highway:False
    parser.add_argument('--with_filter_layer', default=False, help='Utilize filter layer.', action='store_true')
    parser.add_argument('--word_level_MP_dim', type=int, default=-1,
                        help='Number of perspectives for word-level matching.')
    parser.add_argument('--with_match_highway', default=True, help='Utilize highway layers for matching layer.',
                        action='store_true')
    # with_match_highway:False
    # with_aggregation_highway:False
    parser.add_argument('--with_aggregation_highway', default=True,
                        help='Utilize highway layers for aggregation layer.', action='store_true')
    parser.add_argument('--with_lex_decomposition', default=False, help='Utilize lexical decomposition features.',
                        action='store_true')
    parser.add_argument('--lex_decompsition_dim', type=int, default=-1,
                        help='Number of dimension for lexical decomposition features.')
    parser.add_argument('--with_POS', default=False, help='Utilize POS information.', action='store_true')
    parser.add_argument('--with_NER', default=False, help='Utilize NER information.', action='store_true')
    parser.add_argument('--POS_dim', type=int, default=20, help='Number of dimension for POS embeddings.')
    parser.add_argument('--NER_dim', type=int, default=20, help='Number of dimension for NER embeddings.')
    parser.add_argument('--wo_left_match', default=False, help='Without left to right matching.', action='store_true')
    parser.add_argument('--wo_right_match', default=False, help='Without right to left matching', action='store_true')
    parser.add_argument('--wo_full_match', default=False, help='Without full matching.', action='store_true')
    parser.add_argument('--wo_maxpool_match', default=False, help='Without maxpooling matching', action='store_true')
    parser.add_argument('--wo_attentive_match', default=False, help='Without attentive matching', action='store_true')
    parser.add_argument('--wo_max_attentive_match', default=False, help='Without max attentive matching.',
                        action='store_true')
    parser.add_argument('--wo_char', default=True, help='Without character-composed embeddings.', action='store_true')

    #     print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    sys.stdout.flush()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
