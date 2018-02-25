# imports
import torch
import utils
import my_models
import hyperparameters as h # this prints GPU enabled = True


files = ['sample-music.txt', 'input.txt']
# load the inputs as a list of ints
inputs, char2int_cypher, int2char_cypher = utils.load_music(files[0], use_custom=True)
# full input.txt is 501470 in length
dict_size = len(char2int_cypher) # conversion is the dict convert char to int

h.char2int_cypher = char2int_cypher
h.int2char_cypher = int2char_cypher


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

#train(model, optimizer, epochs, train_inputs, validation_set, chunk_size, hidden0)
best_model_dict = my_models.train(lstm, optimizer_lstm, h.epochs, training_set,
                                  validation_set, h.sequence_length, init_hidden)

# save the best model dict to file in case we want to load it later and generate
torch.save(best_model_dict, h.save_file)

my_models.generate('pa4_model', lstm, optimizer_lstm, h.temperature, h.prediction_length)
