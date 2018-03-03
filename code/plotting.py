
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:



## Different Units
loss = np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
plt.figure()
plt.plot(loss[0,:],'--',label='Train')
plt.plot(loss[1,:],label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("LSTM Loss vs Epochs; 100 units, 1 layer, dropout=0, temp=1")
plt.legend()
plt.show()


# In[3]:


## Different Units
loss = np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss2 = np.load('../losses/GRU_1lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
plt.figure()
plt.plot(loss[0,:],'--',label='LSTM Train')
plt.plot(loss[1,:],label='LSTM Validation')
plt.plot(loss2[0,:],'--',label='GRU Train')
plt.plot(loss2[1,:],label='GRU Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("LSTM vs GRU; 100 units, 1 layer, dropout=0, temp=1")
plt.legend()
plt.show()


# In[4]:


## Different Units
loss = np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss1 = np.load('../losses/LSTM_1lay_75unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss2 = np.load('../losses/LSTM_1lay_50unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss3 = np.load('../losses/LSTM_1lay_150unit_25seq_1000batch_100epoch_0drop_1temp.npy')

plt.figure()
plt.plot(loss2[0,:],'--', label='Train 50 units')
plt.plot(loss1[0,:],'--',label='Train 75 units')
plt.plot(loss[0,:],'--',label='Train 100 units')
plt.plot(loss3[0,:],'--', label='Train 150 units')
plt.plot(loss2[1,:], label = 'Validation 50 units')
plt.plot(loss1[1,:],label='Validation 75 units')
plt.plot(loss[1,:],label='Validation 100 units')
plt.plot(loss3[1,:], label = 'Validation 150 units')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("LSTM; Loss with varying hidden units")
plt.legend()
plt.show()


# In[5]:


## Different Units
loss = np.load('../losses/GRU_1lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss1 = np.load('../losses/GRU_1lay_75unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss2 = np.load('../losses/GRU_1lay_50unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss3 = np.load('../losses/GRU_1lay_150unit_25seq_1000batch_100epoch_0drop_1temp.npy')

plt.figure()
plt.plot(loss2[0,:],'--', label='Train 50 units')
plt.plot(loss1[0,:],'--',label='Train 75 units')
plt.plot(loss[0,:],'--',label='Train 100 units')
plt.plot(loss3[0,:],'--', label='Train 150 units')
plt.plot(loss2[1,:], label = 'Validation 50 units')
plt.plot(loss1[1,:],label='Validation 75 units')
plt.plot(loss[1,:],label='Validation 100 units')
plt.plot(loss3[1,:], label = 'Validation 150 units')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("GRU; Loss with varying hidden units")
plt.legend()
plt.show()


# In[6]:


# Different dropout
loss = np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_500epoch_0drop_1temp.npy')
loss1=np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_500epoch_0.1drop_1temp.npy')
loss2 = np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_500epoch_0.2drop_1temp.npy')
loss3 = np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_500epoch_0.3drop_1temp.npy')
plt.figure()
plt.plot(loss[0,:],'--',label='Train 0 dropout')
plt.plot(loss1[0,:],'--',label='Train 0.1 dropout')
plt.plot(loss2[0,:],'--', label='Train 0.2 dropout')
plt.plot(loss3[0,:],'--', label='Train 0.3 dropout')
plt.plot(loss[1,:],label='Validation 0 dropout')
plt.plot(loss1[1,:],label='Validation 0.1 dropout')
plt.plot(loss2[1,:], label = 'Validation 0.2 dropout')
plt.plot(loss3[1,:], label = 'Validation 0.3 dropout')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("LSTM; Loss with varying dropout rates")
plt.legend()
plt.show()


# In[7]:


# Different layers
loss = np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss1=np.load('../losses/LSTM_2lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss2 = np.load('../losses/LSTM_3lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
plt.figure()
plt.plot(loss[0,:],'--',label='Train 1 layer')
plt.plot(loss1[0,:],'--',label='Train 2 layer')
plt.plot(loss2[0,:],'--', label='Train 3 layer')
plt.plot(loss[1,:],label='Validation 1 layer')
plt.plot(loss1[1,:],label='Validation 2 layer')
plt.plot(loss2[1,:], label = 'Validation 3 layer')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("LSTM; loss with varying hidden layers")
plt.legend()
plt.show()


# In[8]:


# Different layers
loss = np.load('../losses/LSTM_3lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss2 = np.load('../losses/GRU_3lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')

plt.figure()
plt.plot(loss[0,:],'--',label='LSTM training')
plt.plot(loss2[0,:],'--', label='GRU training')
plt.plot(loss[1,:],label='LSTM validation')
plt.plot(loss2[1,:], label = 'GRU validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("LSTM vs GRU; loss with 3 hidden layers")
plt.legend()
plt.show()


# In[9]:


# Different Temperatures
loss = np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss1=np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_100epoch_0drop_0.5temp.npy')
loss2 = np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_100epoch_0drop_2temp.npy')
plt.figure()
plt.plot(loss[0,:],'--',label='Train Temp=1')
plt.plot(loss1[0,:],'--',label='Train Temp=0.5')
plt.plot(loss2[0,:],'--', label='Train Temp=2')
plt.plot(loss[1,:],label='Validation Temp=1')
plt.plot(loss1[1,:],label='Validation Temp=0.5')
plt.plot(loss2[1,:], label = 'Validation Temp=2')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("LSTM; Loss with varying temperatures")
plt.legend()
plt.show()


# In[10]:


# Different Temperatures
loss = np.load('../losses/LSTM_1lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss1=np.load('../losses/rms_GRU_1lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
loss2 = np.load('../losses/adagrad_GRU_1lay_100unit_25seq_1000batch_100epoch_0drop_1temp.npy')
plt.figure()
plt.plot(loss[0,:],'--',label='Adam train')
plt.plot(loss1[0,:],'--',label='RMS train')
plt.plot(loss2[0,:],'--', label='Adagrad train')
plt.plot(loss[1,:],label='Adam valid')
plt.plot(loss1[1,:],label='RMS valid')
plt.plot(loss2[1,:], label = 'Adagrad valid')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss with different optimizers")
plt.legend()
plt.show()

