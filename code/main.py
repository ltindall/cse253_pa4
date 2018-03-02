# imports
import torch
import utils
import my_models
import hyperparameters as h # this prints GPU enabled = True


files = ['sample-music.txt', 'input.txt']
# load the inputs as a list of ints
inputs, char2int_cypher, int2char_cypher = utils.load_music(files[1], use_custom=True)
# full input.txt is 501470 in length
dict_size = len(char2int_cypher) # conversion is the dict convert char to int

h.char2int_cypher = char2int_cypher
h.int2char_cypher = int2char_cypher


# define test and validation set
split = int(len(inputs) * h.validation_size) # change 0.1 to how big we want validation set to be
validation_set = inputs[:split]
training_set = inputs[split:]


# create model
lstm = my_models.lstm_char_rnn(dict_size, h.hidden_size, h.num_hidden_layers, batch_size=h.batch_size,dropout_prob=h.dropout,GRU=h.GRU)
init_hidden = lstm.initialize_hidden()
if h.GPU:
    if h.GRU: 
        init_hidden = init_hidden.cuda()
    else:
        print("lstm working")
        init_hidden = (init_hidden[0].cuda(), init_hidden[1].cuda())
    lstm.cuda()

optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=0.01)

mods = my_models.train(lstm, optimizer_lstm, h.epochs, training_set,
                       validation_set, h.sequence_length, init_hidden, force_epochs=True)

best_model_dict = mods[0]
last_model_dict = mods[1]

# save the best model dict to file in case we want to load it later and generate
torch.save(best_model_dict, h.save_file)
torch.save(last_model_dict, h.save_file_progress)
my_models.generate(h.save_file, lstm, h.temperature, h.prediction_length, h.generate_best_file)
my_models.generate(h.save_file_progress, lstm, h.temperature, h.prediction_length, h.generate_last_file)

