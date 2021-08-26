#!/spkr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahulmishra
"""
import numpy as np
from nltk import tokenize

def load_embedding(embedding_file_path, corpus, embedding_dim):
    wordset = set();
    for line in corpus:
        line = line.strip().split()
        for w in line:
            wordset.add(w.lower())
    words_dict = dict(); word_embedding = []; index = 1
    words_dict['$EOF$'] = 0  #add EOF
    word_embedding.append(np.zeros(embedding_dim))
    with open(embedding_file_path, 'r',encoding="utf-8") as f:
        for line in f:
            check = line.strip().split()
            if len(check) == 2: continue
            line = line.strip().split()
            if line[0] not in wordset: continue
            embedding = [float(s) for s in line[1:]]
            word_embedding.append(embedding)
            words_dict[line[0]] = index
            index +=1
    return np.array(word_embedding), words_dict


def fit_transform(x_text, words_dict, max_sen_len, max_doc_len):
    x, sen_len, doc_len = [], [], []
    for index, doc in enumerate(x_text):
        t_sen_len = [0] * max_doc_len
        t_x = np.zeros((max_doc_len, max_sen_len), dtype=int)
        sentences = tokenize.sent_tokenize(doc)
        i = 0
        for sen in sentences:
            j = 0
            for word in sen.strip().split():
                if j >= max_sen_len:
                    break
                if word not in words_dict: continue
                t_x[i, j] = words_dict[word]
                j += 1
            t_sen_len[i] = j
            i += 1
            if i >= max_doc_len:
                break
        doc_len.append(i)
        sen_len.append(t_sen_len)
        x.append(t_x)
    return np.array(x), np.array(sen_len), np.array(doc_len)

class Dataset(object):
    def __init__(self, doc,label, claim, speaker, source,topic):
        self.t_spkr = speaker
        self.t_src = source
        self.t_label = label
        self.t_docs = doc
        self.t_tpc = topic
#        with open(data_file, 'r') as f:
#            for line in f:
#                line = line.strip().decode('utf8', 'ignore').split('\t\t')
#                self.t_spkr.append(line[0])
#                self.t_src.append(line[1])
#                self.t_label.append(int(line[2])-1)
#                self.t_docs.append(line[3].lower())
        self.data_size = len(self.t_docs)

    def get_spkr_dom_tpc_dict(self):
        spkrdict, domdict, tpcdict = dict(), dict(), dict()
        spkridx, domidx, tpcidx = 0, 0, 0
        for u in self.t_spkr:
            if u not in spkrdict:
                spkrdict[u] = spkridx
                spkridx += 1
        for p in self.t_src:
            if p not in domdict:
                domdict[p] = domidx
                domidx += 1
        for t in self.t_tpc:
            if t not in tpcdict:
                tpcdict[t] = tpcidx
                tpcidx += 1     
        if 'Donald Trump' in spkrdict:
            print("found")
        else:
            print("not found")
        return spkrdict, domdict, tpcdict

    def genBatch(self, spkrdict, domdict, tpcdict, wordsdict, batch_size, max_sen_len, max_doc_len, n_class):
        self.epoch = int(len(self.t_docs) / batch_size)
#        self.epoch = len(self.t_docs) 
        if len(self.t_docs) % batch_size != 0:
            self.epoch += 1
        self.spkr = []
        self.dom = []
        self.tpc = []
        self.label = []
        self.docs = []
        self.sen_len = []
        self.doc_len = []
#        try:
#      
#
#            for i in range(self.epoch):
#                self.spkr.append(np.fromiter(map(lambda x: spkrdict[x], self.t_spkr[i*batch_size:(i+1)*batch_size]), dtype=np.int32))
#                self.dom.append(np.fromiter(map(lambda x: domdict[x], self.t_src[i*batch_size:(i+1)*batch_size]), dtype=np.int32))
#                self.label.append(np.eye(n_class, dtype=np.float32)[self.t_label[i*batch_size:(i+1)*batch_size]])
#                b_docs, b_sen_len, b_doc_len = fit_transform(self.t_docs[i*batch_size:(i+1)*batch_size],
#                                                             wordsdict, max_sen_len, max_doc_len)
#                self.docs.append(b_docs)
#                self.sen_len.append(b_sen_len)
#                self.doc_len.append(b_doc_len)
#        except KeyError:
#            pass
        for i in range(self.epoch):
            self.spkr.append(np.fromiter(map(lambda x: spkrdict[x], self.t_spkr[i*batch_size:(i+1)*batch_size]), dtype=np.int32))
            self.dom.append(np.fromiter(map(lambda x: domdict[x], self.t_src[i*batch_size:(i+1)*batch_size]), dtype=np.int32))
            self.tpc.append(np.fromiter(map(lambda x: tpcdict[x], self.t_tpc[i*batch_size:(i+1)*batch_size]), dtype=np.int32))

            self.label.append(np.eye(n_class, dtype=np.float32)[self.t_label[i*batch_size:(i+1)*batch_size]])
            b_docs, b_sen_len, b_doc_len = fit_transform(self.t_docs[i*batch_size:(i+1)*batch_size],
                                                         wordsdict, max_sen_len, max_doc_len)
            self.docs.append(b_docs)
            self.sen_len.append(b_sen_len)
            self.doc_len.append(b_doc_len)
    def batch_iter(self, spkrdict, domdict, tpcdict, wordsdict, n_class, batch_size, num_epochs, max_sen_len, max_doc_len, shuffle=True):
        data_size = len(self.t_docs)
        num_batches_per_epoch = int(data_size / batch_size) + \
                                (1 if data_size % batch_size else 0)
        self.t_spkr = np.array(self.t_spkr)
        self.t_src = np.array(self.t_src)
        self.t_tpc = np.array(self.t_tpc)
        self.t_label = np.array(self.t_label)
        self.t_docs = np.array(self.t_docs)

        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                self.t_spkr = self.t_spkr[shuffle_indices]
                self.t_src = self.t_src[shuffle_indices]
                self.t_tpc = self.t_tpc[shuffle_indices]
                self.t_label = self.t_label[shuffle_indices]
                self.t_docs = self.t_docs[shuffle_indices]

            for batch_num in range(num_batches_per_epoch):
                start = batch_num * batch_size
                end = min((batch_num + 1) * batch_size, data_size)
                spkr = map(lambda x: spkrdict[x], self.t_spkr[start:end])
                dom = map(lambda x: domdict[x], self.t_src[start:end])
                tpc = map(lambda x: tpcdict[x], self.t_tpc[start:end])
                label = np.eye(n_class, dtype=np.float32)[self.t_label[start:end]]
                docs, sen_len, doc_len = fit_transform(self.t_docs[start:end], wordsdict, max_sen_len, max_doc_len)
                batch_data = zip(spkr, dom, tpc, docs, label, sen_len, doc_len)
                yield batch_data


