import torch


# HYPERPARAMETERS 
sequence_length = 25
input_size = output_size = sequence_length
num_hidden_layers = 1
hidden_size = 100
epochs = 2 


GPU = torch.cuda.is_available()
print("GPU enabled = ", GPU)

loss_function = torch.nn.CrossEntropyLoss()
