import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

''' The result obtained with MATHEMATICA '''
# Average Cs_max = 0.497599
# The result of exhaustive search = 1.044 

n_antennas = 1
n_MonteCarlo = 5000
power_levels = [i*0.001 for i in range(0,1000)] #power_levels = [0, 0.1, ..., 0.9]
n_levels = len(power_levels)

avg_Cs_max = 0
avg_CE_max = 0
avg_power_opt = 0

avg_Cs_related = 0

avg_Cs = np.zeros([1,n_levels])
for i in range(n_MonteCarlo):
    print("i = ", i)
    hB = np.random.exponential(1, [1,1])
    hE = np.random.exponential(1, [1,1])
    instantaneous_Cs_max = 0
    instantaneous_CE_max = 0
    Cs_related = 0
    power_opt = 0
    ''' begin of loop '''
    for k in range(n_levels):
        power = power_levels[k]       
        CB = np.log( 1.0 +  20*(hB*power)**2 )  #shape [1,n_levels]
        CE = np.log( 1.0 +  10*(hE*power)**2 ) #shape [1,n_levels]
        Cs = CB-CE
        if CE > 0.5:
            #print("CE > 0.5 and CE = ", CE)
            continue #reject all the remaining statements in the current iteration
                     #then move the control back to the top of the loop
        if instantaneous_CE_max < CE and CE < 0.5:
            instantaneous_CE_max = CE
            power_opt = power
            Cs_related = Cs
            
        if Cs > instantaneous_Cs_max:
            instantaneous_Cs_max = Cs
            
    ''' end of loop '''
    # print("instantaneous_CE_max = ", instantaneous_CE_max)
    # print("power_opt = ", power_opt)
    # print("Cs_related = ", Cs_related)
    # print("instantaneous_Cs_max = ", instantaneous_Cs_max)
    avg_Cs_max = avg_Cs_max + instantaneous_Cs_max
    avg_CE_max = avg_CE_max + instantaneous_CE_max
    avg_power_opt = avg_power_opt + power_opt 
    
    avg_Cs_related = avg_Cs_related + Cs_related
    
    
print("============================")
avg_Cs_max = avg_Cs_max/n_MonteCarlo
avg_CE_max = avg_CE_max/n_MonteCarlo 
avg_power_opt = avg_power_opt/n_MonteCarlo 
avg_Cs_related = avg_Cs_related/n_MonteCarlo 
print("avg_Cs_max = ", avg_Cs_max)
print("avg_Cs_related = ", avg_Cs_related)
print("avg_CE_max = ", avg_CE_max)
print("avg_power_opt = ", avg_power_opt)

