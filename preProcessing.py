## coding=utf8
## word embbedding的数据预处理
import re
import numpy as np


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))  # ceil: 返回不小于x的最小整数。  batch=3178
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(nb_batch)]


def pad_2d_matrix(word_idx_batch, max_length=None, dtype=np.int32):
    if max_length is None: max_length = np.max([len(cur_in_val) for cur_in_val in word_idx_batch])
    batch_size = len(word_idx_batch)
    out_val = np.zeros((batch_size, max_length), dtype=dtype)

    for i in iter(range(batch_size)):
        cur_in_val = word_idx_batch[i]
        kept_length = len(cur_in_val)
        if kept_length > max_length: kept_length = max_length
        out_val[i, :kept_length] = cur_in_val[:kept_length]
    return out_val


class DataStream(object):
    def __init__(self, inpath, word_vocab=None, label_vocab=None, batch_size=60, isShuffle=False, isLoop=False,
                 isSort=True, max_sent_length=200):
        ## section 1
        instances = []
        for line in open(inpath, "r"):
            line = line.strip()
            items = re.split("\t", line)
            label = items[0]
            sentence1 = items[1].lower()
            sentence2 = items[2].lower()
            if label_vocab is not None:
                label_id = label_vocab.getIndex(label)  # 得到id
                if label_id >= label_vocab.vocab_size: label_id = 0
            else:
                label_id = int(label)
            word_idx_1 = word_vocab.to_index_sequence(sentence1)  # 都是根据字典来得到的一个list[3,5,2,6,3...]
            word_idx_2 = word_vocab.to_index_sequence(sentence2)
            if len(word_idx_1) > max_sent_length:
                word_idx_1 = word_idx_1[:max_sent_length]
            if len(word_idx_2) > max_sent_length:
                word_idx_2 = word_idx_2[:max_sent_length]
            instances.append((label, sentence1, sentence2, label_id, word_idx_1, word_idx_2))
        if isSort: instances = sorted(instances, key=lambda instance: (
            len(instance[4]), len(instance[5])))  # sort instances based on length
        self.num_instances = len(instances)

        ## section 2
        batch_spans = make_batches(self.num_instances, batch_size)  # list[(0,60),(60,120),(...)]
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            label_batch = []
            sent1_batch = []
            sent2_batch = []
            label_id_batch = []
            word_idx_1_batch = []
            word_idx_2_batch = []
            sent1_length_batch = []
            sent2_length_batch = []

            ##  word_idx_1: sentence1的list表示
            for i in iter(range(batch_start, batch_end)):
                (label, sentence1, sentence2, label_id, word_idx_1, word_idx_2) = instances[i]
                label_batch.append(label)
                sent1_batch.append(sentence1)
                sent2_batch.append(sentence2)
                label_id_batch.append(label_id)
                word_idx_1_batch.append(word_idx_1)
                word_idx_2_batch.append(word_idx_2)
                sent1_length_batch.append(len(word_idx_1))
                sent2_length_batch.append(len(word_idx_2))

            cur_batch_size = len(label_batch)
            if cur_batch_size == 0:
                continue

            # padding
            max_sent1_length = np.max(sent1_length_batch)
            max_sent2_length = np.max(sent2_length_batch)

            # 每个句子的截断
            label_id_batch = np.array(label_id_batch)
            word_idx_1_batch = pad_2d_matrix(word_idx_1_batch, max_length=max_sent1_length)
            word_idx_2_batch = pad_2d_matrix(word_idx_2_batch, max_length=max_sent2_length)

            sent1_length_batch = np.array(sent1_length_batch)
            sent2_length_batch = np.array(sent2_length_batch)

            self.batches.append(
                (label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch,
                 sent1_length_batch, sent2_length_batch))

        instances = None
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)  # 数据打乱顺序
        self.isLoop = isLoop
        self.cur_pointer = 0


    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch
