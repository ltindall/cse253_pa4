# CSE 253 PA4 

# imports
import torch
import torchvision
from torch.autograd import Variable
import numpy as np
import utils




### Pytorch definitions 

loss_function = torch.nn.CrossEntropyLoss()

GPU = torch.cuda.is_available()
print("GPU enabled = ",GPU)


class lstm_char_rnn(torch.nn.Module):
  def __init__(self, dict_size, hidden_size, num_hidden_layers):
    super(lstm_char_rnn, self).__init__()

    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers

    self.encoder = torch.nn.Embedding(dict_size, hidden_size)
    self.recurrent = torch.nn.GRU(hidden_size, hidden_size, num_hidden_layers)
    self.decoder = torch.nn.Linear(hidden_size, dict_size)

  def forward(self, inputs, hidden):
    embedding_output = self.encoder(inputs)
    recurrent_output, hidden = self.recurrent(embedding_output, hidden)
    output = self.decoder(recurrent_output.view(recurrent_output.size(0) * recurrent_output.size(1), recurrent_output.size(2)))
    
    return output, hidden

  def initialize_hidden(self):
    return Variable(torch.zeros(self.num_hidden_layers, 1, self.hidden_size))



'''
### LSTM MODEL 
def lstm_char_rnn(input_size, hidden_size, num_hidden_layers, output_size):
  # save model state with model.state_dict(), load the model state with model.load_state_dict()
  
  model = torch.nn.Sequential(
              torch.nn.Embedding(input_size, hidden_size), 
              torch.nn.LSTM(hidden_size, hidden_size, num_hidden_layers), 
              torch.nn.Linear(hidden_size, output_size)     
          )
  
  if GPU:
    model = model.cuda()
  return model
'''

### Generic train method 
def train(model, optimizer, epochs, train_inputs): 

  # get a list of start indices based on the inputs
  pm_arr = utils.permute_list(len(train_inputs), sequence_length+1)
  batch = []
  while(len(pm_arr) > 0): 
    b,ids = utils.create_batch(train_inputs, pm_arr, 1, sequence_length+1)
    batch.append(list(b.reshape(sequence_length+1)))
  batch = np.array(batch)
  print(batch.shape)
  
  for i in range(epochs): 
    print("training epoch ",i)

    j = 0
    for seq in batch: 
      if j %10000 == 0: 
        print("seq ",j)
      j = j +  1
    
      if torch.equal(hidden0.data, hcpy.data):
        print('failed')
    
      model.train(True)

      # split batch into inputs and targets
      inputs = Variable(torch.LongTensor(seq[:-1])).view(sequence_length,-1)
      targets = Variable(torch.LongTensor(seq[1:]))
      if GPU: 
        inputs = inputs.cuda()
        targets = targets.cuda()

      outputs,_ = model(inputs,hidden0)

      loss = loss_function(outputs, targets)

      optimizer.zero_grad()
            
      loss.backward()
      
      optimizer.step()

  torch.save(model.state_dict(), 'pa4_model')

  return 1
  

# HYPERPARAMETERS 
sequence_length = 25
input_size = output_size = sequence_length
num_hidden_layers = 1
hidden_size = 100
epochs = 10 


# PREPROCESSING 
#text = load_music()

#converted_text = list(map(ord, list(text)))

# load the inputs as a list of ints
inputs, cypher = utils.load_music('input.txt') # full input.txt is 501470 in length

dict_size = len(cypher)


#batch_size = int(len(inputs)/sequence_length)
mini_batch_size = 1000

'''
# get a list of start indices based on the inputs
pm_arr = utils.permute_list(len(inputs), sequence_length+1)
# create a batch based on the permute array (do this with a while loop in the epoch)
#batch, ids = utils.create_batch(inputs, pm_arr, mini_batch_size, sequence_length+1)
batch = []
while(len(pm_arr) > 0): 
  b,ids = utils.create_batch(inputs, pm_arr, 1, sequence_length+1)
  batch.append(list(b.reshape(sequence_length+1)))
batch = np.array(batch)
print(batch.shape)
'''



# LSTM model 
lstm = lstm_char_rnn(dict_size,hidden_size, num_hidden_layers)
hidden0 = lstm.initialize_hidden()

if GPU: 
  hidden0 = hidden0.cuda()
  lstm.cuda()
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr = 0.01)

import copy
hcpy = hidden0.clone()

train(lstm, optimizer_lstm, epochs, inputs)
