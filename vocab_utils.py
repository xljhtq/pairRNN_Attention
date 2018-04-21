# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import re
import os


# import math
class Vocab(object):
    def __init__(self, pattern):
        self.pattern = pattern

    def patternWord(self, train_path="", model_dir=""):
        vec_path = train_path
        self.word2id = {}
        self.id2word = {}

        vec_file = open(vec_path, 'rt')
        word_vecs = {}
        for line in vec_file:
            line = line.strip()
            parts = line.split(' ')
            word = parts[0]
            self.word_dim = len(parts[1:])
            if self.word_dim < 128: continue
            vector = np.array(parts[1:], dtype='float32')
            cur_index = len(self.word2id)
            self.word2id[word] = cur_index
            self.id2word[cur_index] = word
            word_vecs[cur_index] = vector
        vec_file.close()
        cur_index = len(self.word2id)
        self.word2id['<UNK/>'] = cur_index
        self.id2word[cur_index] = '<UNK/>'
        word_vecs[cur_index] = np.random.normal(0, 1, size=(self.word_dim,))
        self.vocab_size = len(self.word2id)

        self.word_vecs = np.zeros((self.vocab_size, self.word_dim),
                                  dtype=np.float32)
        for cur_index in iter(range(self.vocab_size)):
            self.word_vecs[cur_index][:len(word_vecs[cur_index])] = word_vecs[cur_index]

        word2id_path = model_dir + "/word2id.txt"
        print("word2id path:", word2id_path)
        with open(word2id_path, "w") as out_op:
            for word in self.word2id:
                out_op.write(word + "\t" + str(self.word2id[word]) + "\n")

    def patternLabel(self, voc, label_path=""):
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = len(voc)  # voc=all_chars
        self.word_dim = 2
        for word in voc:
            cur_index = len(self.word2id)
            self.word2id[word] = cur_index
            self.id2word[cur_index] = word

        shape = (self.vocab_size, self.word_dim)
        scale = 0.05
        self.word_vecs = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32)

        with open(label_path, "w") as out_op:
            for id in self.id2word:
                wordVec = ""
                for v in self.word_vecs[id]:
                    wordVec += "{} ".format(v)
                out_op.write(str(id) + "\t" + self.id2word[id] + "\t" + wordVec + "\n")

    def to_index_sequence(self, sentence):
        sentence = sentence.strip()
        seq = []
        for word in re.split('\\s+', sentence):
            if word in self.word2id:
                idx = self.word2id[word]
            else:
                idx = self.word2id['<UNK/>']
            seq.append(idx)
        return seq
