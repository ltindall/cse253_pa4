import torch
import torchvision
from torch.autograd import Variable
import numpy as np
import utils
import hyperparameters as h


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


### Generic Train Function
def train(model, optimizer, epochs, train_inputs, sequence_length, hidden0): 
  # get a list of start indices based on the inputs
  pm_list = utils.permute_list(len(train_inputs), sequence_length+1)
  
  for i in range(epochs): 
    print("training epoch ",i)

    j = 0
    while len(pm_list) > 0: 
      if j %10000 == 0: 
        print("seq ",j)
      j = j + 1
      model.train(True)
      
      start_id = pm_list.pop()
      end_id = start_id+sequence_length

      # split batch into inputs and targets
      inputs = Variable(torch.LongTensor(train_inputs[start_id:end_id])).view(sequence_length, -1)
      targets = Variable(torch.LongTensor(train_inputs[start_id+1:end_id+1]))
      if h.GPU: 
        inputs = inputs.cuda()
        targets = targets.cuda()

      outputs,_ = model(inputs, hidden0)

      loss = h.loss_function(outputs, targets)

      optimizer.zero_grad()
            
      loss.backward()
      
      optimizer.step()

  torch.save(model.state_dict(), 'pa4_model')

  return 1