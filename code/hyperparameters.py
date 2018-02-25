import torch


### HYPERPARAMETERS
sequence_length = 40
input_size = output_size = sequence_length
num_hidden_layers = 1
hidden_size = 100
epochs = 300
batch_size = 1000
temperature = 1
prediction_length = 500
stop_criterion = 4 # if loss increases 3 times in a row
overlap_data = False


### Start and end substitution for better learning
use_custom_startend = False
start_sub = '$'
end_sub = ';'
start_orig = '<start>'
end_orig = '<end>'


### Pytorch stuff
GPU = torch.cuda.is_available()
print("GPU is {}enabled ".format(['not ', ''][GPU]))

loss_function = torch.nn.CrossEntropyLoss()


### model saving file paths
save_file = 'pa4_model.txt'
# if gpu crashes before we finish when we're doing many many epochs
save_file_progress = 'pa4_model_before_stop.txt'
