# Deep Reinforcement Learning-based Distributed Dynamic Spectrum Access in Multi-user Multi-channel Cognitive Radio Internet of Things Networks
   author={Zhang, Xiaohui; Chen, Ze; Zhang, Yinghui; Liu, Yang; Jin, Minglu; Qiu, Tianshuang}
   journal= {IEEE Internet of Things Journal}


### Introduction
This project includes:
[DSA_env.py]: Define two classes, DSA_Markov and DSA_Period, which model different scenarios of DSA in a wireless communication environment. These classes are part of a simulation for evaluating the performance of Spectrum Sharing strategies.
[PRDDQN.py]: The setup and training process of the PRD-DQN with prioritized experience replay using Echo State Networks.
[pyESN_online.py]: The RNN structure for ESN.


### How the code works
[main.py]: Run this file, inclduing setting system parameters. Firstly, initialize the environment, including number of channel and su. Secondly, create a new folder in the result folder according to the environment configuration. For example, when the number of channels is 12 and the number of su is 7, you can create a folder named "channel_12_su_7_1" in the result folder to store the training data. Finally, adjust the save path through the code "file_folder = '.\\result\\channel_%d_su_%d_*' % (n_channel, n_su)". Note that you only need to change the "*" according to the folder created in the previous step. For example, according to the previous step, the code can be modified to "file_folder = '.\\result\\channel_%d_su_%d_1' % (n_channel, n_su)".


### Required packages
-Tensorflow 1.x. or Tensorflow 2.x. 
-numpy
-matplotlib
The versions of Python and libraries used in our work: Python = 3.7, Tensorflow = 2.7.0, numpy = 1.21.6, matplotlib = 3.5.3.


### Remark
The experimental results of this article are based on Monte Carlo simulation results generated from random data, and the simulation data is presented in our paper "Deep Reinforcement Learning-based Distributed Dynamic Spectrum Access in Multi-user Multi-channel Cognitive Radio Internet of Things Networks".