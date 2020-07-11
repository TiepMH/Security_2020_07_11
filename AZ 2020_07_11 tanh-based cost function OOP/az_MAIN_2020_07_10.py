import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Tiep_Library import Norm_of_a_vector, squaredNorm_of_a_vector
from Tiep_Library import Norm_of_uH_times_v, squaredNorm_of_uH_times_v
from Tiep_Library import Norm_of_xH_times_A, squaredNorm_of_xH_times_A
from class_Networks_WL import NeuralNetwork_WL 

n_antennas = 1
size_in = n_antennas
size_out = n_antennas

learning_rate = 0.001
n_hidden = 10

size_batch = 256
n_Iterations = 10

n_epochs = 301

''' Declare objects '''
object_NN_WoongLee = NeuralNetwork_WL("Method1", 
                                   size_in, size_out, learning_rate, n_hidden)

''' Prepare containters to save values '''
#At each iteration, we want to save some corresponding values 
cost_array = np.zeros([1, n_epochs]) #An array of costs 
p_array = np.zeros([1, n_epochs]) 
CB_array = np.zeros([1, n_epochs]) 
CE_array = np.zeros([1, n_epochs]) 
Cs_array = np.zeros([1, n_epochs]) 
g_pos_array = np.zeros([1, n_epochs]) # [g]+ = max(0,g) where g<=0 is the constraint 

#After all iterations are finished, we want to save the cumulative values
cum_cost = np.zeros([1, n_epochs]) #Cumulative value of cost 
cum_p = np.zeros([1, n_epochs]) #Cumulative value of power
cum_CB = np.zeros([1, n_epochs]) #Cumulative value of C_B 
cum_CE = np.zeros([1, n_epochs]) #Cumulative value of C_E
cum_Cs = np.zeros([1, n_epochs]) #Cumulative value of C_s
cum_g_pos = np.zeros([1, n_epochs]) #Cumulative value of [g]+ = max(0, g) 


''' Run Session '''
with tf.Session() as sess:
    for iteration in range(n_Iterations):
        sess.run(tf.global_variables_initializer())
        print("Iteration = ",iteration)
        
        hB_val = np.random.exponential(1, [size_batch, n_antennas])
        hE_val = np.random.exponential(1, [size_batch, n_antennas])
        h_val = np.concatenate((hB_val,hE_val), axis=1)     
        
        ''' TRAINING '''
        for epoch in range(n_epochs):
            object_NN_WoongLee.train(hB_val, hE_val, sess) #Training
            cost_val = object_NN_WoongLee.get_cost(hB_val, hE_val, sess) #Cost
            p_val = object_NN_WoongLee.get_output(hB_val, hE_val, sess) #Power 
            CB_val = object_NN_WoongLee.get_CB(hB_val, hE_val, sess) #Bob's capacity
            CE_val = object_NN_WoongLee.get_CE(hB_val, hE_val, sess) #Eve's capacity
            Cs_val = object_NN_WoongLee.get_Cs(hB_val, hE_val, sess) #Secrecy rate
            g_pos_val = object_NN_WoongLee.get_g_pos(hB_val, hE_val, sess) #Constraint violation

            # Calculate the average values over a batch of multiple examples
            p_array[0][epoch] = np.mean(p_val)
            cost_array[0][epoch] = np.mean(cost_val)    
            CB_array[0][epoch] = np.mean(CB_val) ##
            CE_array[0][epoch] = np.mean(CE_val) ##
            Cs_array[0][epoch] = np.mean(Cs_val) ##
            g_pos_array[0][epoch] = np.mean(g_pos_val)            
        ''' End of training '''
        
        cum_p = cum_p + p_array 
        cum_cost = cum_cost + cost_array
        cum_CB = cum_CB + CB_array
        cum_CE = cum_CE + CE_array
        cum_Cs = cum_Cs + Cs_array 
        cum_g_pos = cum_g_pos + g_pos_array
        
    ''' End of iterations '''
    avg_p_array = cum_p/n_Iterations
    avg_cost_array = cum_cost/n_Iterations
    avg_CB_array = cum_CB/n_Iterations ##
    avg_CE_array = cum_CE/n_Iterations ##
    avg_Cs_array = cum_Cs/n_Iterations ##
    avg_g_pos_array = cum_g_pos/n_Iterations  

###############################################################################
''' Save data for later use '''
np.save("avg_CB_array_WL.npy", avg_CB_array)
np.save("avg_CE_array_WL.npy", avg_CE_array)
np.save("avg_Cs_array_WL.npy", avg_Cs_array)
np.save("avg_p_array_WL.npy", avg_p_array)
np.save("avg_g_array_WL.npy", avg_g_pos_array)
    

''' Show results '''
iteration = [i for i in range(len(avg_Cs_array[0]))]
Cs_opt_search = [1.044 for _ in range(len(avg_Cs_array[0]))]

plt.figure(1)
plt.plot(iteration, avg_Cs_array[0], label="Method 1", color='b', linewidth=2)
plt.plot(iteration, Cs_opt_search, label="Exhaustive search", color='k', linestyle='--', linewidth=2)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Average $C_s$', fontsize=15)
#plt.title('Figure 1')
plt.legend(loc='lower right', fontsize=12)

plt.figure(2)
plt.plot(iteration, avg_CB_array[0], label="$C_B$", color='b', linewidth=2)
plt.plot(iteration, avg_CE_array[0], label="$C_E$", color='r', linewidth=2)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel(r'Average capacities' , fontsize=15)
#plt.title('Figure 1')
plt.legend(loc='upper right', fontsize=12)

plt.figure(3)
plt.plot(iteration, avg_p_array[0], label="Method 1", color='b', linewidth=2)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Average power', fontsize=15)
#plt.title('Figure 1')
plt.legend(loc='upper right')

plt.figure(4)
plt.plot(iteration, avg_g_pos_array[0], label="Method 1", color='b', linewidth=2)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Average $[g]^+$', fontsize=15)
#plt.title('Figure 1')
plt.legend(loc='upper right')