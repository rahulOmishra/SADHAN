#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahulmishra
"""
import os, time, pickle
import numpy as np
import tensorflow as tf
import datetime
import data_helpers1
from data_helpers1 import Dataset
from model1 import SADHA

import argparse
#from tensorflow import reset_default_graph
args = argparse.ArgumentParser()

config, unparsed = args.parse_known_args()


os.environ["CUDA_VISIBLE_DEVICES"]="0"

#reset_default_graph()
# Data loading params
args.add_argument("n_class", type=int, default=2)
args.add_argument("-dataset", default='politi')
# Model Hyperparameters
args.add_argument("embedding_dim", type=int, default=100)
args.add_argument("hidden_size", type=int, default=100)
args.add_argument('max_sen_len',type=int, default=500)
args.add_argument('max_doc_len',type=int,default= 40)
args.add_argument("lr",type=float, default=0.001)

# Training parameters
args.add_argument("batch_size",type=int, default=100)
args.add_argument("num_epochs", type=int,default=10)
args.add_argument("evaluate_every",type=int, default=50)

# Misc Parameters
args.add_argument("allow_soft_placement",type= bool,default= True)
args.add_argument("log_device_placement",type= bool, default=False)

config1, unparsed = args.parse_known_args()#import sys
#FLAGS(sys.argv)
#FLAGS._parse_flags()
print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items()):
#    print("{}={}".format(attr.upper(), value))
#print("")


# Load data
print("Loading data...")


data_dir = "/home/prosjekt/deepnews/fakenews-data/snopes_pickles/"
with open(data_dir+'docVectorCollection.pickle', 'rb') as handle:
    doc_data = pickle.load(handle)
with open(data_dir+'labelVectorCollection.pickle', 'rb') as handle:
    label_data = pickle.load(handle)
with open(data_dir+'claimVectorCollection.pickle', 'rb') as handle:
    claim_data = pickle.load(handle)
with open(data_dir+'SpeakerVectorCollection.pickle', 'rb') as handle:
    speaker_data = pickle.load(handle)
with open(data_dir+'DocSourceCollection.pickle', 'rb') as handle:
    source_doc_data = pickle.load(handle)
with open(data_dir+'TopicVectorCollection.pickle', 'rb') as handle:
    topic_data = pickle.load(handle)
#experi
#source_doc_data = topic_data
#experi
train_doc = doc_data[:int((len(doc_data)+1)*.80)]
train_label = label_data[:int((len(doc_data)+1)*.80)]
train_claim = claim_data[:int((len(doc_data)+1)*.80)]
train_speaker = speaker_data[:int((len(doc_data)+1)*.80)]
train_source_doc = source_doc_data[:int((len(doc_data)+1)*.80)]
train_topic = topic_data[:int((len(doc_data)+1)*.80)]


test_doc = doc_data[int(len(train_doc)*.80+1):]
test_label = label_data[int(len(train_doc)*.80+1):]
test_claim = claim_data[int(len(train_doc)*.80+1):]
test_speaker = speaker_data[int(len(train_doc)*.80+1):]
test_source_doc = source_doc_data[int(len(train_doc)*.80+1):]
test_topic = topic_data[int(len(train_doc)*.80+1):]


Ftrain_doc = train_doc[:int((len(train_doc)+1)*.80)]
Ftrain_label = train_label[:int((len(train_doc)+1)*.80)]
Ftrain_claim = train_claim[:int((len(train_doc)+1)*.80)]
Ftrain_speaker = train_speaker[:int((len(train_doc)+1)*.80)]
Ftrain_source_doc = train_source_doc[:int((len(train_doc)+1)*.80)]
Ftrain_topic = train_topic[:int((len(train_doc)+1)*.80)]


val_doc = train_doc[int(len(train_doc)*.80+1):]
val_label = train_label[int(len(train_doc)*.80+1):]
val_claim = train_claim[int(len(train_doc)*.80+1):]
val_speaker = train_speaker[int(len(train_doc)*.80+1):]
val_source_doc = train_source_doc[int(len(train_doc)*.80+1):]
val_topic = train_topic[int(len(train_doc)*.80+1):]

#print(val_doc,val_label,val_claim,val_speaker,val_source_doc)
all_d = Dataset(doc_data, label_data, claim_data, speaker_data, source_doc_data, topic_data)
trainset = Dataset(Ftrain_doc, Ftrain_label, Ftrain_claim, Ftrain_speaker, Ftrain_source_doc, Ftrain_topic)
devset = Dataset(val_doc, val_label, val_claim, val_speaker, val_source_doc, val_topic)
testset = Dataset(test_doc, test_label, test_claim, test_speaker, test_source_doc, test_topic)
print(devset.t_label[0])

alldata = np.concatenate([trainset.t_docs, devset.t_docs, testset.t_docs], axis=0)
#print(alldata)
embeddingpath = '/home/prosjekt/deepnews/falseclaims-data/clean_data/glove.6B.100d.txt'
embeddingfile, wordsdict = data_helpers1.load_embedding(embeddingpath, alldata, config1.embedding_dim)
#del alldata
print("Loading data finished...")

spkrdict, domdict, tpcdict = all_d.get_spkr_dom_tpc_dict()
trainbatches = trainset.batch_iter(spkrdict, domdict, tpcdict, wordsdict, config1.n_class, config1.batch_size,
                                 config1.num_epochs, config1.max_sen_len, config1.max_doc_len)
#spkrdict, domdict = devset.get_spkr_dom_dict()

devset.genBatch(spkrdict, domdict, tpcdict, wordsdict, config1.batch_size,
                  config1.max_sen_len, config1.max_doc_len, config1.n_class)
#spkrdict, domdict = testset.get_spkr_dom_dict()

testset.genBatch(spkrdict, domdict, tpcdict, wordsdict, config1.batch_size,
                  config1.max_sen_len, config1.max_doc_len, config1.n_class)


#devbatches = devset.batch_iter(spkrdict, domdict, wordsdict, FLAGS.n_class, FLAGS.batch_size,
#                                 FLAGS.num_epochs, FLAGS.max_sen_len, FLAGS.max_doc_len)
#testbatches = testset.batch_iter(spkrdict, domdict, wordsdict, FLAGS.n_class, FLAGS.batch_size,
#                                 FLAGS.num_epochs, FLAGS.max_sen_len, FLAGS.max_doc_len)

with tf.Graph().as_default():
    session_config = tf.ConfigProto(
            
        allow_soft_placement=config1.allow_soft_placement,
        log_device_placement=config1.log_device_placement
    )
    session_config.gpu_options.allow_growth = False
    session_config.gpu_options.allocator_type = 'BFC'

#    config = tf.ConfigProto(device_count = {'GPU': 1})
    sess = tf.Session(config=session_config)
    with sess.as_default():
        SADHA = SADHA(
            max_sen_len = config1.max_sen_len,
            max_doc_len = config1.max_doc_len,
            class_num = config1.n_class,
            embedding_file = embeddingfile,
            embedding_dim = config1.embedding_dim,
            hidden_size = config1.hidden_size,
            spkr_num = len(spkrdict),
            dom_num = len(domdict),
            tpc_num = len(tpcdict)
        )
        SADHA.build_model()
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(config1.lr)
        grads_and_vars = optimizer.compute_gradients(SADHA.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        def datasetfrac(dataset):
            datafrac = 1.1 / dataset.data_size
            return datafrac
        # Save dict
        timestamp = str(int(time.time()))
        checkpoint_dir = os.path.abspath("../checkpoints/"+config1.dataset+"/"+timestamp)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        with open(checkpoint_dir + "/wordsdict.txt", 'wb') as f:
            pickle.dump(wordsdict, f)
        with open(checkpoint_dir + "/spkrdict.txt", 'wb') as f:
            pickle.dump(spkrdict, f)
        with open(checkpoint_dir + "/domdict.txt", 'wb') as f:
            pickle.dump(domdict, f)
        with open(checkpoint_dir + "/tpcdict.txt", 'wb') as f:
            pickle.dump(tpcdict, f)

        sess.run(tf.global_variables_initializer())

        def train_step(batch):
            u, p, t, x, y, sen_len, doc_len = zip(*batch)
            feed_dict = {
                SADHA.spkrid: u,
                SADHA.domid: p,
                SADHA.tpcid: t,
                SADHA.input_x: x,
                SADHA.input_y: y,
                SADHA.sen_len: sen_len,
                SADHA.doc_len: doc_len
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, SADHA.loss, SADHA.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def predict_step(u, p, t, x, y, sen_len, doc_len, name=None):
            feed_dict = {
                SADHA.spkrid: u,
                SADHA.domid: p,
                SADHA.tpcid: t,
                SADHA.input_x: x,
                SADHA.input_y: y,
                SADHA.sen_len: sen_len,
                SADHA.doc_len: doc_len
            }
            step, loss, accuracy, correct_num, mse = sess.run(
                [global_step, SADHA.loss, SADHA.accuracy, SADHA.correct_num, SADHA.mse],
                feed_dict)
            return correct_num, accuracy, mse

        def predict(dataset, name=None):
            acc = 0
            rmse = 0.
#            print("speaker  ",dataset.spkr[0])
            for i in range(dataset.epoch):
                if ((i+1)*100) < len(dataset.t_docs): 
#                    print("value of i ",i)
                    correct_num, _, mse = predict_step(dataset.spkr[i], dataset.dom[i], dataset.tpc[i], dataset.docs[i],
                                                       dataset.label[i], dataset.sen_len[i], dataset.doc_len[i], name)
                    acc += correct_num
                    rmse += mse
            
            rmse = np.sqrt(rmse * datasetfrac(dataset))
            return acc, rmse

        topacc = 0.
        toprmse = 0.
        better_dev_acc = 0.
        predict_round = 0

        # Training loop. For each batch...
        for tr_batch in trainbatches:
            train_step(tr_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % config1.evaluate_every == 0:
                predict_round += 1
                print("\nEvaluation round %d:" % (predict_round))

                dev_acc, dev_rmse = predict(devset, name="dev")
                print("dev_acc: %.4f    dev_RMSE: %.4f" % (dev_acc, dev_rmse))
                test_acc, test_rmse = predict(testset, name="test")
                print("test_acc: %.4f    test_RMSE: %.4f" % (test_acc, test_rmse))

#                print topacc with best dev acc
                if dev_acc >= better_dev_acc:
                    better_dev_acc = dev_acc
                    topacc = test_acc
                    toprmse = test_rmse
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                print("topacc: %.4f   RMSE: %.4f" % (topacc, toprmse))
