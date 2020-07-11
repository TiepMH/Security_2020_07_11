import tensorflow as tf
import numpy as np

def Norm_of_a_vector(u_row):
    # u_row = [ [u_1, u_2, ... , u_N]     example 1
    #           [u_1, u_2, ... , u_N]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_N] ]   example ?
    norm_of_vector = tf.norm(u_row, axis=1)
    # norm_of_vector = [ [ (u_1**2 + u_2_**2 + ... u_N**2)^0.5 ]   ex1
    #                    [ (u_1**2 + u_2_**2 + ... u_N**2)^0.5 ]   ex2
    #                       ..................................
    #                    [ (u_1**2 + u_2_**2 + ... u_N**2)^0.5 ] ] exN
    return tf.reshape(norm_of_vector, [tf.size(norm_of_vector),1])

def squaredNorm_of_a_vector(u_row):
    # u_row = [ [u_1, u_2, ... , u_N]     example 1
    #           [u_1, u_2, ... , u_N]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_N] ]   example ?
    squared_u_element_wise = u_row**2
    # squared_u_element_wise = [ [u_1**2, u_2**2, ... , u_N**2]     example 1
    #                            [u_1**2, u_2**2, ... , u_N**2]     example 2
    #                             ... ,    ... ,  ... , ...
    #                            [u_1**2, u_2**2, ... , u_N**2] ]   example ?
    squaredNorm_of_vector = tf.reduce_sum(squared_u_element_wise, 
                                          axis=1, keepdims=True)
    #squaredNorm_of_vector = [ [u_1**2 + u_2**2 + ... + u_N**2]   example 1
    #                          [u_1**2 + u_2**2 + ... + u_N**2]   example 2
    #                           ..............................
    #                          [u_1**2 + u_2**2 + ... + u_N**2]   example ?
    return squaredNorm_of_vector

def Norm_of_uH_times_v(u_row, v_row):
    # u_row = [ [u_1, u_2, ... , u_N]     example 1
    #           [u_1, u_2, ... , u_N]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_N] ]   example ?
    #################################################
    # v_row = [ [v_1, v_2, ... , v_N]     example 1
    #           [v_1, v_2, ... , v_N]     example 2
    #            ..., ..., ... , ...
    #           [v_1, v_2, ... , v_N] ]   example ?
    #################################################
    uv_element_wise = tf.multiply(u_row,v_row)
    # uv_element_wise = [ [u_1 v_1, u_2 v_2, ... , u_N v_N]     example 1
    #                     [u_1 v_1, u_2 v_2, ... , u_N v_N]     example 2
    #                      ..., ..., ... , ...
    #                     [u_1 v_1, u_2 v_2, ... , u_N v_N] ]   example ?
    #################################################
    uH_times_v = tf.reduce_sum(uv_element_wise, axis=1, keepdims=True)
    # uH_times_v = [ [u_1 v_1 + u_2 v_2 + ... + u_N v_N]     example 1
    #                [u_1 v_1 + u_2 v_2 + ... + u_N v_N]     example 2
    #                ..................................
    #                [u_1 v_1 + u_2 v_2 + ... + u_N v_N] ]   example ?
    # shape [?, 1] where ? is the size of a batch
    #################################################
    uH_times_v_norm = tf.norm(uH_times_v) 
    # squaredNorm = [ [ a_1**2 ]     example 1
    #                 [ a_2**2 ]     example 2
    #                  ...
    #                 [ a_N**2 ] ]   example ?
    return uH_times_v_norm

def squaredNorm_of_uH_times_v(u_row, v_row):
    # u_row = [ [u_1, u_2, ... , u_N]     example 1
    #           [u_1, u_2, ... , u_N]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_N] ]   example ?
    #################################################
    # v_row = [ [v_1, v_2, ... , v_N]     example 1
    #           [v_1, v_2, ... , v_N]     example 2
    #           ..., ..., ... , ...
    #           [v_1, v_2, ... , v_N] ]   example ?
    #################################################
    uv_element_wise = tf.multiply(u_row,v_row)
    # uv_element_wise = [ [u_1 v_1, u_2 v_2, ... , u_N v_N]     example 1
    #                     [u_1 v_1, u_2 v_2, ... , u_N v_N]     example 2
    #                      ..., ..., ... , ...
    #                     [u_1 v_1, u_2 v_2, ... , u_N v_N] ]   example ?
    #################################################
    uH_times_v = tf.reduce_sum(uv_element_wise, axis=1, keepdims=True)
    # uH_times_v = [ [u_1 v_1 + u_2 v_2 + ... + u_N v_N]     example 1
    #                [u_1 v_1 + u_2 v_2 + ... + u_N v_N]     example 2
    #                ..................................
    #                [u_1 v_1 + u_2 v_2 + ... + u_N v_N] ]   example ?
    # shape [?, 1] where ? is the size of a batch
    #################################################
    squaredNorm = uH_times_v**2 
    # squaredNorm = [ [ a_1**2 ]     example 1
    #                 [ a_2**2 ]     example 2
    #                  ...
    #                 [ a_N**2 ] ]   example ?
    return squaredNorm

def Norm_of_xH_times_A(x_row,A):
    # x_row = [ [u_1, u_2, ... , u_M]     example 1
    #           [u_1, u_2, ... , u_M]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_M] ]   example ?
    #################################################
    # A = [ [a11, a12, ..., a1N]
    #       [a21, a22, ..., a2N]
    #       ..., ..., ... , ...
    #       [aM1, aM2, ..., aMN]
    # shape [M,N]
    #################################################
    xH_times_Ax_row_vector = tf.matmul(x_row,A) 
    # xH_times_Ax_row_vector = [ [u_1, u_2, ..., u_M] A    ==>> example 1
    #                            [u_1, u_2, ..., u_M] A    ==>> example 2
    #                             ..., ..., ..., ...     
    #                            [u_1, u_2, ..., u_M] A ]  ==>> example ?
    # 
    #                       = [ [h_1, h_2, ..., h_N]    example 1
    #                           [h_1, h_2, ..., h_N]    example 2
    #                            ..., ..., ..., ...
    #                           [h_1, h_2, ..., h_N] ]  example ?
    #################################################
    xA_norm = tf.norm(xH_times_Ax_row_vector,axis=1)
    return tf.reshape(xA_norm, [tf.size(xA_norm),1])

def squaredNorm_of_xH_times_A(x_row,A):
    # x_row = [ [u_1, u_2, ... , u_M]     example 1
    #           [u_1, u_2, ... , u_M]     example 2
    #            ..., ..., ... , ...
    #           [u_1, u_2, ... , u_M] ]   example ?
    #################################################
    # A = [ [a11, a12, ..., a1N]
    #       [a21, a22, ..., a2N]
    #       ..., ..., ... , ...
    #       [aM1, aM2, ..., aMN]
    # shape [M,N]
    #################################################
    xH_times_Ax_row_vector = tf.matmul(x_row,A) 
    # xH_times_Ax_row_vector = [ [u_1, u_2, ..., u_M] A    ==>> example 1
    #                            [u_1, u_2, ..., u_M] A    ==>> example 2
    #                             ..., ..., ..., ...     
    #                            [u_1, u_2, ..., u_M] A ]  ==>> example ?
    # 
    #                       = [ [h_1, h_2, ..., h_N]    example 1
    #                           [h_1, h_2, ..., h_N]    example 2
    #                            ..., ..., ..., ...
    #                           [h_1, h_2, ..., h_N] ]  example ?
    #################################################
    squared_xH_times_A_element_wise = tf.abs(xH_times_Ax_row_vector)**2
    #squared_xH_times_A_element_wise = [ [h_1**2, h_2**2, ..., h_N**2]    example 1
    #                                    [h_1**2, h_2**2, ..., h_N**2]    example 2
    #                                      ...  ,  ... ,  ...,  ...
    #                                    [h_1**2, h_2**2, ..., h_N**2] ]  example ?
    #################################################
    squaredNorm = tf.reduce_sum(squared_xH_times_A_element_wise, axis=1)
    # squaredNorm = [ [h_1**2 + h_2**2 + ...+ h_N**2]    example 1
    #                 [h_1**2 + h_2**2 + ...+ h_N**2]    example 2
    #                   .........................
    #                 [h_1**2 + h_2**2 + ...+ h_N**2] ]  example ?
    #
    #             = [ [ ||h||^2 ]     example 1
    #                 [ ||h||^2 ]     example 2
    #                 ..........
    #                 [ ||h||^2 ] ]   example ?
    return tf.reshape(squaredNorm, [tf.size(squaredNorm),1])
    