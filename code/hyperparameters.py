import torch


### HYPERPARAMETERS
sequence_length = 50
input_size = output_size = sequence_length
num_hidden_layers = 2
hidden_size = 128
epochs = 300
batch_size = 1000
temperature = 1
stop_criterion = 4 # if loss increases 3 times in a row

validation_size = 0.2 # fraction of data to use as validation set



### function options
overlap_data = False

GRU = True


### Start and end substitution for better learning
use_custom_startend = False
start_sub = '$'
end_sub = ';'
start_orig = '<start>'
end_orig = '<end>'

prediction_length = 600
till_end = False
map_width = 20

### Pytorch stuff
GPU = torch.cuda.is_available()
print("GPU is {}enabled ".format(['not ', ''][GPU]))

loss_function = torch.nn.CrossEntropyLoss()


### model saving file paths
save_file = 'pa4_model.txt'
# if gpu crashes before we finish when we're doing many many epochs
save_file_progress = 'pa4_model_before_stop.txt'
generate_best_file = 'best_music.txt'
generate_last_file = 'last_music.txt'
