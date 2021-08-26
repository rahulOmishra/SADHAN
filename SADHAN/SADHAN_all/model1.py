#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahulmishra
"""
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

class SADHA(object):

    def __init__(self, max_sen_len, max_doc_len, class_num, embedding_file,
            embedding_dim, hidden_size, spkr_num, dom_num, tpc_num):
        self.max_sen_len = max_sen_len
        self.max_doc_len = max_doc_len
        self.class_num = class_num
        self.embedding_file = embedding_file
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.spkr_num = spkr_num
        self.dom_num = dom_num
        self.tpc_num = tpc_num

        with tf.name_scope('input'):
            self.spkrid = tf.placeholder(tf.int32, [None], name="speaker_id")
            self.domid = tf.placeholder(tf.int32, [None], name="domain_id")
            self.tpcid = tf.placeholder(tf.int32, [None], name="topic_id")           
            self.input_x = tf.placeholder(tf.int32, [None, self.max_doc_len, self.max_sen_len], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, self.class_num], name="input_y")
            self.sen_len = tf.placeholder(tf.int32, [None, self.max_doc_len], name="sen_len")
            self.doc_len = tf.placeholder(tf.int32, [None], name="doc_len")

        with tf.name_scope('weights'):
            self.weights = {
                'softmax': tf.Variable(tf.random_uniform([6 * self.hidden_size, self.class_num], -0.01, 0.01)),

                'u_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'u_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'u_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                'u_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'u_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                'p_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'p_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'p_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                'p_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'p_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                
                't_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                't_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                't_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                't_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                't_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                'wu_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'wp_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'wt_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'wu_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'wp_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'wt_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),

            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),

                'u_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'u_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                'u_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                'p_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'p_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                'p_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                
                't_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                't_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                't_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
            }

        with tf.name_scope('embedding'):
            self.word_embedding = tf.constant(self.embedding_file, name="word_embedding", dtype=tf.float32)
            self.x = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
            self.speaker_embedding = tf.Variable(tf.random_uniform([self.spkr_num, 2*self.hidden_size], -0.01, 0.01), dtype=tf.float32, trainable=False)
            self.domain_embedding = tf.Variable(tf.random_uniform([self.dom_num, 2*self.hidden_size], -0.01, 0.01), dtype=tf.float32, trainable=False)
            self.topic_embedding = tf.Variable(tf.random_uniform([self.tpc_num, 2*self.hidden_size], -0.01, 0.01), dtype=tf.float32, trainable=False)

            self.speaker = tf.nn.embedding_lookup(self.speaker_embedding, self.spkrid)
            self.domain = tf.nn.embedding_lookup(self.domain_embedding, self.domid)
            self.topic = tf.nn.embedding_lookup(self.topic_embedding, self.tpcid)



    def softmax(self, inputs, length, max_length):
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.exp(inputs)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
        return inputs / _sum


    def speaker_attention(self):
        inputs = tf.reshape(self.x, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('u_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                cell_bw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='u_word'
            )
            outputs = tf.concat(outputs,2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('u_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['u_wh_1']) + self.biases['u_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.speaker, self.weights['wu_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['u_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('u_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                cell_bw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='u_sentence'
            )
            outputs = tf.concat(outputs,2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('u_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['u_wh_2']) + self.biases['u_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.speaker, self.weights['wu_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['u_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('u_softmax'):
            self.u_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.u_scores = tf.matmul(self.u_doc, self.weights['u_softmax']) + self.biases['u_softmax']
            self.u_predictions = tf.argmax(self.u_scores, 1, name="u_predictions")

        with tf.name_scope("u_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.u_scores, labels = self.input_y)
            self.u_loss = tf.reduce_mean(losses)

        with tf.name_scope("u_accuracy"):
            correct_predictions = tf.equal(self.u_predictions, tf.argmax(self.input_y, 1))
            self.u_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.u_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="u_accuracy")


    def Domain_attention(self):
        inputs = tf.reshape(self.x, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('p_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
#                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0, activation = tf.nn.selu),
#                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0, activation = tf.nn.selu),
                cell_fw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                cell_bw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='p_word'
            )
            outputs = tf.concat(outputs,2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('p_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['p_wh_1']) + self.biases['p_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.domain, self.weights['wp_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['p_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('p_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
#                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0, activation = tf.nn.selu),
#                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0, activation = tf.nn.selu),
                cell_fw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                cell_bw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='p_sentence'
            )
            outputs = tf.concat(outputs,2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('p_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['p_wh_2']) + self.biases['p_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.domain, self.weights['wp_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['p_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('p_softmax'):
            self.p_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.p_scores = tf.matmul(self.p_doc, self.weights['p_softmax']) + self.biases['p_softmax']
            self.p_predictions = tf.argmax(self.p_scores, 1, name="p_predictions")

        with tf.name_scope("p_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.p_scores, labels = self.input_y)
            self.p_loss = tf.reduce_mean(losses)

        with tf.name_scope("p_accuracy"):
            correct_predictions = tf.equal(self.p_predictions, tf.argmax(self.input_y, 1))
            self.p_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.p_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="p_accuracy")
            
            
    def topic_attention(self):
        inputs = tf.reshape(self.x, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('t_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
#                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0, activation = tf.nn.selu),
#                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0, activation = tf.nn.selu),
                cell_fw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                cell_bw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='t_word'
            )
            outputs = tf.concat(outputs,2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('t_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['t_wh_1']) + self.biases['t_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.topic, self.weights['wt_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['t_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('t_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
#                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0, activation = tf.nn.selu),
#                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0, activation = tf.nn.selu),
                cell_fw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                cell_bw=tf.nn.rnn_cell.GRUCell(self.hidden_size,  activation = tf.nn.selu),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='t_sentence'
            )
            outputs = tf.concat(outputs,2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('t_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['t_wh_2']) + self.biases['t_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.topic, self.weights['wt_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['t_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('t_softmax'):
            self.t_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.t_scores = tf.matmul(self.t_doc, self.weights['t_softmax']) + self.biases['t_softmax']
            self.t_predictions = tf.argmax(self.t_scores, 1, name="t_predictions")

        with tf.name_scope("t_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.t_scores, labels = self.input_y)
            self.t_loss = tf.reduce_mean(losses)

        with tf.name_scope("t_accuracy"):
            correct_predictions = tf.equal(self.t_predictions, tf.argmax(self.input_y, 1))
            self.t_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.t_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="t_accuracy")

    def build_model(self):
        self.speaker_attention()
        self.Domain_attention()
        self.topic_attention()

        with tf.name_scope('softmax'):
            outputs = tf.concat([self.u_doc, self.p_doc, self.t_doc],1)
#            outputs = tf.concat([self.p_doc, self.t_doc],1)
            self.scores = tf.matmul(outputs, self.weights['softmax']) + self.biases['softmax']
#            self.scores = tf.matmul(self.u_doc, self.weights['u_softmax']) + self.biases['u_softmax']
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
            self.loss = 0.7*tf.reduce_mean(losses) + 0.1*self.u_loss  + 0.1*self.p_loss + 0.1*self.t_loss
#            self.loss =  tf.reduce_mean(losses) 


        with tf.name_scope("metrics"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.mse = tf.reduce_sum(tf.square(self.predictions - tf.argmax(self.input_y, 1)), name="mse")
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32), name="correct_num")
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")