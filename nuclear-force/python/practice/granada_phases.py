import numpy as np

Tlabs = [1,5,10,25,50,100,150,200,250,300,350]

#Granada phase shifts for Tlab = 1,5,10,25,50,100,150,200,250,300,350
#1S0
delta_1S0 = np.array([62.07659,63.66045,60.02014,51.07080,40.88493,27.21011,17.10006,8.83047,1.97071,-3.61817,-8.04425])
delta_1S0_errors = np.array([0.01832,0.04398,0.06260,0.09867,0.14734,0.23050,0.26576,0.26739,0.28110,0.34394,0.45311])
#3S1 - 3D1
delta_3S1 = np.array([147.64721,117.95350,102.29182,80.13580,62.16019, 42.71247, 30.39210, 20.93573, 12.94459,  5.83733, -0.64397])
delta_3S1_errors = np.array([0.01008,0.02214,0.03053,0.04423,0.05252,0.05269,0.05763,0.07173,0.08665,0.09804,0.10513])
delta_3D1 = np.array([-0.00494,-0.18001,-0.67115,-2.78785,-6.42312,-12.19832,-16.40696,-19.68207,-22.32437,-24.36879,-25.69867])
delta_3D1_errors = np.array([0.00001,0.00046,0.00182,0.00918,0.02555,0.04849,0.05783,0.07148,0.08047,0.10789,0.19051])
delta_3E1  = np.array([0.10420,0.65668,1.12535,1.73387,2.05660,2.47261,2.93809,3.44815,4.03636,4.75330,5.61036])
delta_3E1_errors = np.array([0.00053,0.00422,0.00891,0.02045,0.03668,0.06754,0.09491,0.11237,0.11072,0.11264,0.16939])