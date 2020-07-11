import tensorflow as tf
import numpy as np
from Tiep_Library import Norm_of_a_vector, squaredNorm_of_a_vector
from Tiep_Library import Norm_of_uH_times_v, squaredNorm_of_uH_times_v
from Tiep_Library import Norm_of_xH_times_A, squaredNorm_of_xH_times_A

class NeuralNetwork_WL:
    
    def __init__(self, name, size_in, size_out, learning_rate, n_hidden):
        self.size_in = size_in
        self.size_out = size_out
        self.learning_rate = learning_rate 
        self.n_hidden = n_hidden
        
        self.hB = tf.placeholder(tf.float32, [None, self.size_in])
        self.hE = tf.placeholder(tf.float32, [None, self.size_in])
        self.input, self.output = self.architecture(name)
            
        ''' Compute some important expressions '''
        # Compute CB
        self.squaredNorm_of_hBH_and_p = squaredNorm_of_uH_times_v(self.hB, self.output)
        self.CB = tf.log( tf.add(1.0, 20*self.squaredNorm_of_hBH_and_p ) )
        # Compute CE
        self.squaredNorm_of_hEH_and_p = squaredNorm_of_uH_times_v(self.hE, self.output)
        self.CE = tf.log( tf.add(1.0, 10*self.squaredNorm_of_hEH_and_p ) )
        # Compute Cs
        self.Cs = tf.subtract(self.CB, self.CE)
        # Contraint g<=0
        self.g = tf.subtract(self.CE, 0.5)
        self.g_pos = tf.maximum(self.g, 0.0) # [g]+ = max(0,g)
        
        ###
        self.lambDa_1 = 0.5
        self.lambDa_2 = 1 - self.lambDa_1 
        self.delta = 1
        
        ''' Cost function corresponding to ONLY ONE example '''
        self.cost_WL = tf.add( self.lambDa_1*(- self.Cs),
                                self.lambDa_2*tf.tanh(self.g_pos/self.delta) 
                                )
        
        ''' Cost function for training is the average cost over a batch of examples '''
        self.cost_of_a_batch = tf.reduce_mean(self.cost_WL)
        
        ''' Optimizer and training operation '''
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.cost_of_a_batch)
        
    def architecture(self, name):
        net_in = tf.concat([self.hB,self.hE], axis=1)
        #net_in = tf.placeholder(tf.float32, [None, self.size_in])
        net = tf.layers.dense(net_in, self.n_hidden, activation=tf.nn.relu, use_bias=True,
                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                      bias_initializer=tf.contrib.layers.xavier_initializer())
        net = tf.layers.dense(net, self.n_hidden, activation=tf.nn.relu, use_bias=True,
                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                      bias_initializer=tf.contrib.layers.xavier_initializer())
        net = tf.layers.dense(net, self.n_hidden, activation=tf.nn.relu, use_bias=True,
                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                      bias_initializer=tf.contrib.layers.xavier_initializer())
        net = tf.layers.dense(net, self.n_hidden, activation=tf.nn.relu, use_bias=True,
                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                       bias_initializer=tf.contrib.layers.xavier_initializer())
        net_out = tf.layers.dense(net, self.size_out, activation=tf.nn.sigmoid,
                                  kernel_initializer = tf.random_normal_initializer(),
                                  bias_initializer = tf.constant_initializer(1))
        return net_in, net_out 
    
    def train(self, hB_val, hE_val, sess):
        return sess.run(self.train_op, feed_dict={self.hB: hB_val, self.hE: hE_val})
    
    def get_output(self, hB_val, hE_val, sess):
        return sess.run(self.output, feed_dict={self.hB: hB_val, self.hE: hE_val})
    
    def get_cost(self, hB_val, hE_val, sess):
        return sess.run(self.cost_WL, feed_dict={self.hB: hB_val, self.hE: hE_val})
    
    def get_CB(self, hB_val, hE_val, sess):
        return sess.run( self.CB, feed_dict={self.hB: hB_val, self.hE: hE_val})
    
    def get_CE(self, hB_val, hE_val, sess):
        return sess.run( self.CE, feed_dict={self.hB: hB_val, self.hE: hE_val})
    
    def get_Cs(self, hB_val, hE_val, sess):
        return sess.run( self.Cs, feed_dict={self.hB: hB_val, self.hE: hE_val})
    
    def get_g_pos(self, hB_val, hE_val, sess):
        return sess.run( self.g_pos, feed_dict={self.hB: hB_val, self.hE: hE_val})