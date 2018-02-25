
# coding: utf-8

# In[ ]:


# imports
import torch, torchvision
import numpy as np
import utils
import my_models
import hyperparameters as h # this prints GPU enabled = True


# load the inputs as a list of ints
inputs, char2int_cypher = utils.load_music('input.txt')
# full input.txt is 501470 in length
dict_size = len(char2int_cypher) # conversion is the dict convert char to int


# define test and validation set
split = int(len(inputs) * 0.1) # change 0.1 to how big we want validation set to be
validation_set = inputs[:split]
training_set = inputs[split:]


# create model
lstm = my_models.lstm_char_rnn(dict_size, h.hidden_size, h.num_hidden_layers, batch_size=h.batch_size)
init_hidden = lstm.initialize_hidden()
if h.GPU:
    init_hidden = init_hidden.cuda()
    lstm.cuda()

optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=0.01)


# In[4]:





# In[ ]:


my_models.train(lstm, optimizer_lstm, h.epochs, inputs, h.sequence_length, init_hidden)

