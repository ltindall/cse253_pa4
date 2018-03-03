import torch


### HYPERPARAMETERS
sequence_length = 25
input_size = output_size = sequence_length
num_hidden_layers = 1
hidden_size = 100
epochs = 100
batch_size = 2000
temperature = 1
stop_criterion = 4 # if loss increases 3 times in a row

validation_size = 0.2 # fraction of data to use as validation set

dropout = 0

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

rnn_type = 'GRU' if GRU else 'LSTM'
### model saving file paths

save_string = ('adam_'+rnn_type+'_'+str(num_hidden_layers)+'lay_'+str(hidden_size)+'unit_'
            +str(sequence_length)+'seq_'+str(batch_size)+'batch_'+str(epochs)
            +'epoch_'+str(dropout)+'drop_'+str(temperature)+'temp')

save_file = ('bestmodel_'+save_string+'.txt')
# if gpu crashes before we finish when we're doing many many epochs
save_file_progress = ('lastmodel_'+save_string+'.txt')
generate_best_file = ('bestmusic_'+save_string+'.txt')
generate_last_file = ('lastmusic_'+save_string+'.txt')
