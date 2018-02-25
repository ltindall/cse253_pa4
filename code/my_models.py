import torch
from torch.autograd import Variable
import numpy as np
import utils
import hyperparameters as h
import os


class lstm_char_rnn(torch.nn.Module):
    def __init__(self, dict_size, hidden_size, num_hidden_layers, batch_size):
        super(lstm_char_rnn, self).__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size

        self.encoder = torch.nn.Embedding(dict_size, hidden_size)
        self.recurrent = torch.nn.GRU(hidden_size, hidden_size, num_hidden_layers)
        self.decoder = torch.nn.Linear(hidden_size, dict_size)

    def forward(self, inputs, hidden):
        embedding_output = self.encoder(inputs)
        recurrent_output, hidden = self.recurrent(embedding_output, hidden)
        output = self.decoder(recurrent_output.view(recurrent_output.size(0)*recurrent_output.size(1),
                                                    recurrent_output.size(2)))

        return output, hidden

    def initialize_hidden(self):
        return Variable(torch.zeros(self.num_hidden_layers, self.batch_size,
                                    self.hidden_size))


### Generic Train Function
def train(model, optimizer, epochs, train_inputs, chunk_size, hidden0):
    # get a list of start indices based on the inputs
    batch_size = model.batch_size

    for i in range(epochs):
        print("training epoch ",i)
        pm_list = utils.permute_list(len(train_inputs), chunk_size+1, overlap=True)


        # there is no do-while loop in python...
        j = 0
        while True:
            # create minibatch
            batch, batch_targets, ids = utils.create_batch(train_inputs, pm_list,
                                                           batch_size, chunk_size)
            if batch is None:
                # end of the epoch, no more data to train on
                print('finished with this epoch!')
                break

            if j %10000 == 0:
                print("seq ",j)

            j += 1
            model.train(True)

            # split batch into inputs and targets
            inputs = Variable(torch.LongTensor(batch))
            targets = Variable(torch.LongTensor(batch_targets)).contiguous().view(-1)
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


### Generic train method
def generate(model_filepath, model, optimizer, train_inputs, temperature, prediction_length):

    if os.path.exists(model_filepath):
        model.load_state_dict(torch.load(model_filepath))
    else:
        raise ValueError('Model file not found.')

    # zero out hidden weights to prime the network
    model.batch_size = 1
    generate_hidden = model.initialize_hidden()

    if h.GPU:
        generate_hidden = generate_hidden.cuda()

    model.train(False)

    # prime the network
    first_chars = '<start>'
    for ch in first_chars:
        input_char = Variable(torch.LongTensor([h.char2int_cypher[ch]])).view(1,-1)
        if h.GPU:
            input_char = input_char.cuda()

        output, generate_hidden = model(input_char, generate_hidden)


    output_char = first_chars[-1]
    output_int = h.char2int_cypher[output_char]
    predicted_chars = first_chars
    for i in range(prediction_length):
        input_char = Variable(torch.LongTensor([output_int])).view(1, -1)
        if h.GPU:
            input_char = input_char.cuda()

        output, generate_hidden = model(input_char, generate_hidden)

        softmax_dist = torch.nn.functional.softmax(output/temperature)
        output_int = int(torch.multinomial(softmax_dist,1)[0])

        output_char = h.int2char_cypher[output_int]
        predicted_chars += output_char


    print("final output = ",predicted_chars)
    return 1

