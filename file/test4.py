# 给数据加指定SNR的高斯噪声
import torch
import numpy as np

SNR = 5
noise = np.random.randn(signal.shape[0],signal.shape[1]) 	#产生N(0,1)噪声数据
noise = noise-np.mean(noise) 								#均值为0
signal_power = np.linalg.norm( signal - signal.mean() )**2 / signal.size	#此处是信号的std**2
noise_variance = signal_power/np.power(10,(SNR/10))         #此处是噪声的std**2
noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    ##此处是噪声的std**2
signal_noise = noise + signal

Ps = ( np.linalg.norm(signal - signal.mean()) )**2          #signal power
Pn = ( np.linalg.norm(signal - signal_noise ) )**2          #noise power
snr = 10*np.log10(Ps/Pn)
