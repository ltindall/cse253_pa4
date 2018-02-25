import torch


# HYPERPARAMETERS
sequence_length = 50
input_size = output_size = sequence_length
num_hidden_layers = 2
hidden_size = 100
epochs = 40
batch_size = 500
temperature = 1
prediction_length = 500

GPU = torch.cuda.is_available()
print("GPU enabled = ", GPU)

loss_function = torch.nn.CrossEntropyLoss()
