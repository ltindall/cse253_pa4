import torch
from torch.autograd import Variable
import numpy as np
import utils
import hyperparameters as h


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
        pm_list = utils.permute_list(len(train_inputs), chunk_size+1, overlap=False)


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