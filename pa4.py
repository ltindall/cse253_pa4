# CSE 253 PA4 

import torch
import torchvision
from torch.autograd import Variable
import numpy as np



loss_function = nn.CrossEntropyLoss()

GPU = torch.cuda.is_available()
print("GPU enabled = ",GPU)




# MUSIC LOADER 
def load_music(): 
  with open("./data/input.txt", 'r') as f:
    text = f.read()
  return text



# LSTM MODEL 
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


def train(model, optimizer, epochs, train_data): 



  for i in range(epochs): 

    for batch in batches: 
      model.train(True)


      if GPU: 
        batch = batch.cuda()


      # split batch into inputs and targets

      outputs = model(inputs)

      loss = loss_function(outputs, targets)

      optimizer.zero_grad()
            
      loss.backward()
      
      optimizer.step()

  return 1
  


# PREPROCESSING 
text = load_music()

converted_text = list(map(ord, list(text)))
  

# HYPERPARAMETERS 
sequence_length = 25
input_size = output_size = sequence_length
num_hidden_layers = 1
hidden_size = 100
epochs = 10 



# LSTM model 
lstm = lstm_char_rnn(input_size, hidden_size, num_hidden_layers, output_size)
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr = 0.01)




