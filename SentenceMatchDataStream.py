# coding=utf8
import numpy as np
import re


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))  # ceil: 返回不小于x的最小整数。  batch=3178
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in
            range(0, nb_batch)]  # zgwang: starting point of each batch


def pad_2d_matrix(in_val, max_length=25, word_vocab=None):
    batch_size = len(in_val)
    padded_sentences = []
    for i in iter(range(batch_size)):
        currentSentence = in_val[i]
        new_sentence = currentSentence
        kept_length = len(currentSentence)
        if kept_length <= max_length:
            num_padding = max_length - kept_length
            new_sentence = currentSentence + [word_vocab.word2id['<UNK/>']] * num_padding
        padded_sentences.append(new_sentence)
    return np.array(padded_sentences)


class SentenceMatchDataStream(object):
    def __init__(self, inpath, word_vocab=None, label_vocab=None,
                 batch_size=60, isShuffle=False, isLoop=False, isSort=False, max_sent_length=25):
        instances = []  # 存储转换的list值
        infile = open(inpath, 'rt')
        for cnt, line in enumerate(infile):
            line = line.strip()
            if line.startswith('-'): continue
            items = re.split("\t", line)
            label = items[0]
            sentence1 = items[1].lower()
            sentence2 = items[2].lower()
            if label_vocab is not None:
                label_id = label_vocab.word2id[label]  # 得到id
            else:
                label_id = int(label)
            word_idx_1 = word_vocab.to_index_sequence(sentence1)
            word_idx_2 = word_vocab.to_index_sequence(sentence2)

            if len(word_idx_1) > max_sent_length:
                word_idx_1 = word_idx_1[:max_sent_length]
            if len(word_idx_2) > max_sent_length:
                word_idx_2 = word_idx_2[:max_sent_length]

            instances.append((label_id, word_idx_1, word_idx_2))

        infile.close()

        print("---------isSort-----------", isSort)
        if isSort: instances = sorted(instances,
                                      key=lambda instance: (
                                          len(instance[1]), len(instance[2])))  # sort instances based on length
        self.num_instances = len(instances)

        # distribute into different buckets
        batch_spans = make_batches(self.num_instances, batch_size)  # list[(0,60),(60,120),(...)]
        self.batches = []  # 第2个self
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            label_id_batch = []
            word_idx_1_batch = []
            word_idx_2_batch = []
            sent1_length_batch = []
            sent2_length_batch = []

            ##  word_idx_1: sentence1的list表示
            for i in iter(range(batch_start, batch_end)):
                (label_id, word_idx_1, word_idx_2) = instances[i]

                label_id_batch.append(label_id)
                word_idx_1_batch.append(word_idx_1)
                word_idx_2_batch.append(word_idx_2)
                sent1_length_batch.append(max_sent_length)
                sent2_length_batch.append(max_sent_length)

            cur_batch_size = len(label_id_batch)
            if cur_batch_size == 0: continue

            # padding
            label_id_batch = np.array(label_id_batch)
            word_idx_1_batch = pad_2d_matrix(word_idx_1_batch, max_length=max_sent_length, word_vocab=word_vocab)
            word_idx_2_batch = pad_2d_matrix(word_idx_2_batch, max_length=max_sent_length, word_vocab=word_vocab)

            sent1_length_batch = np.array(sent1_length_batch)
            sent2_length_batch = np.array(sent2_length_batch)

            self.batches.append(
                (label_id_batch, word_idx_1_batch, word_idx_2_batch, sent1_length_batch, sent2_length_batch))

        instances = None
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)  # 数据打乱顺序
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
            # print('{} '.format(self.index_array[self.cur_pointer]))
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]
