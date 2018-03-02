import torch
from torch.autograd import Variable
import utils
import hyperparameters as h
import copy
import os
import numpy as np
import matplotlib.pyplot as plt

class lstm_char_rnn(torch.nn.Module):
    def __init__(self, dict_size, hidden_size, num_hidden_layers, batch_size,dropout_prob=0,GRU=True):
        super(lstm_char_rnn, self).__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size

        self.encoder = torch.nn.Embedding(dict_size, hidden_size)
        self.drop = torch.nn.Dropout(dropout_prob)

        if GRU:
            self.recurrent = torch.nn.GRU(hidden_size, hidden_size, num_hidden_layers,dropout=dropout_prob)
        else:
            self.recurrent = torch.nn.LSTM(hidden_size, hidden_size, num_hidden_layers,dropout=dropout_prob)
        self.decoder = torch.nn.Linear(hidden_size, dict_size)

    def forward(self, inputs, hidden):
        embedding_output = self.encoder(inputs)
        recurrent_output, hidden = self.recurrent(embedding_output, hidden)
        output = self.decoder(recurrent_output.view(recurrent_output.size(0)*recurrent_output.size(1),
                                                    recurrent_output.size(2)))

        return output, hidden, recurrent_output

    def initialize_hidden(self):

        if h.GRU: 
            return Variable(torch.zeros(self.num_hidden_layers, self.batch_size,
                                    self.hidden_size))
        else: 
            return (Variable(torch.zeros(self.num_hidden_layers, self.batch_size, self.hidden_size)), 
                    Variable(torch.zeros(self.num_hidden_layers, self.batch_size, self.hidden_size)))


### Generic Train Function
def train(model, optimizer, epochs, train_set, validation_set, chunk_size,
          hidden0, force_epochs=False):
    """
    Generic train function
    input:
        <torch obj> model: the model we want to train
        <torch obj> optimizer: an optimizer from pytorch
        <int> epochs: number of epochs
        <list> train_set: the traing set, list of ints
        <list> validation_set: validation set in same format as train_set
        <int> chunk_size: the length of a sequence (number of time steps)
        <torch obj> hidden0: the initial hidden layer to send to recurrent layer
        <bool> force_epochs: option to ignore early stopping and train on more epochs
                             has some memory optimizations
    return:
        <OrderedDict> the weights of the model that performed the best on the
                      validation set.
    """
    # get a list of start indices based on the inputs
    batch_size = model.batch_size

    phases = ['training', 'validation']
    sets = (train_set, validation_set)

    # store the losses in a dict of lists (special case if we force all epochs)
    if force_epochs:
        losses = {x: [0 for a in range(h.stop_criterion)] for x in phases}
    else:
        losses = {x: [] for x in phases} # store everything
    best_loss = None
    stop_flag = False
    best_model_state = copy.deepcopy(model.state_dict())

    for i in range(epochs):
        print('\nEpoch {}/{}'.format(i, epochs - 1))
        print('-' * 10)
        # it shouldn't matter if we shuffle validation set after each epoch...
        # would be more work if I tried to keep it constant
        validation_indices = utils.permute_list(len(validation_set),
                                                chunk_size+1, overlap=h.overlap_data)
        permuted_train_indices = utils.permute_list(len(train_set),
                                                    chunk_size+1, overlap=h.overlap_data)
        for phase, phase_name in enumerate(phases):
            # some preprocessing to set parameters for training or validation
            data = sets[phase]

            if phase_name == 'training':
                # scheduler step would go here
                constructor_list = permuted_train_indices
                model.train(True)
            else:
                constructor_list = validation_indices
                model.train(False)

            # we only need loss
            running_loss = 0.0
            batch_num = 0

            # there is no do-while loop in python...
            while True:
                # create minibatch
                batch, batch_targets, ids = utils.create_batch(data, constructor_list,
                                                               batch_size, chunk_size)
                if batch is None:
                    # end of the epoch, no more data to train on
                    print('finished {} with this epoch!'.format(phase_name))
                    break

                # debug code here (progress)
                if batch_num%1000 == 0:
                    print("Working on sequence {}-{}".format(batch_size * batch_num, 
                                                             batch_size * (batch_num+batch_size)))
                batch_num += 1

                # split batch into inputs and targets
                inputs = Variable(torch.LongTensor(batch))
                targets = Variable(torch.LongTensor(batch_targets)).contiguous().view(-1)
                if h.GPU:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                # sketch but i guess it works? Reset for each new sequence batch
                outputs,_,_ = model(inputs, hidden0)
                 
                loss = h.loss_function(outputs/h.temperature, targets)

                optimizer.zero_grad()
                # backprop, and optimize
                if phase_name == 'training':
                    loss.backward() # propagate the error
                    optimizer.step()

                # normalize loss by the batchsize * chunk_size per batch
                running_loss += loss.data[0]

            # calculate the epoch loss, normalized by the number of batches
            losses[phase_name].append(running_loss/batch_num)
            '''
            if not force_epochs:
                losses[phase_name].append(running_loss/batch_num)
            else:
                losses[phase_name] = utils.shift_list(losses[phase_name],
                                                      running_loss/batch_num)
            '''                                

            print('{} Loss:\t{}\n'.format(phase_name, losses[phase_name][-1]))
            # validation phase model saving, and early stopping
            if phase_name == 'validation':
                # set the best loss to the first loss if we've never set it
                if best_loss is None:
                    best_loss = losses[phase_name][-1]

                # save model state when we encounter a new best loss
                if losses[phase_name][-1] < best_loss:
                    best_model_state = copy.deepcopy(model.state_dict())

                    if force_epochs:
                        # save the best model state too
                        torch.save(best_model_state, h.save_file_progress)
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_loss = losses[phase_name][-1]

                # if we're forcing epochs, then we ignore early stopping
                # Do early stopping if we have enough epochs to start
                if not force_epochs and i >= h.stop_criterion:
                    splice = losses[phase_name][-h.stop_criterion:]
                    if utils.monotonic_increase(splice):
                        stop_flag = True

        last_model_state = copy.deepcopy(model.state_dict())
        if stop_flag:
            # early stopping, get out of the epochs loop too
            print('Validation set Loss increased {} times in a row!'.format(h.stop_criterion))
            print('Best validation loss: {}\n'.format(best_loss))
            print('Reverting model to best model so far...')
            model.load_state_dict(best_model_state)
            print('Finished reverting the model')
            
            
            # TODO plot the stuff
            plt.plot(np.trim_zeros(np.array(losses[phases[0]])),label='Train')
            plt.plot(np.trim_zeros(np.array(losses[phases[1]])),label='Validation')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            break
    plt.plot(np.trim_zeros(np.array(losses[phases[0]])),label='Train')
    plt.plot(np.trim_zeros(np.array(losses[phases[1]])),label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.show()
    

    return best_model_state, last_model_state


### Generic train method
def generate(model_state, model, temperature, prediction_length, generate_file, till_end=True ):


    if os.path.exists(model_state):
        model.load_state_dict(torch.load(model_state))
    else:
        raise ValueError('Model file not found.')

    
    # zero out hidden weights to prime the network
    model.batch_size = 1
    generate_hidden = model.initialize_hidden()

    if h.GPU:
        if h.GRU: 
            generate_hidden = generate_hidden.cuda()
        else: 
            generate_hidden = (generate_hidden[0].cuda(), generate_hidden[1].cuda())

    model.train(False)

    # prime the network
    if h.use_custom_startend:
        first_chars = h.start_sub + '\nX'
    else:
        first_chars = h.start_orig + '\nX'

    for ch in first_chars:
        input_char = Variable(torch.LongTensor([h.char2int_cypher[ch]])).view(1,-1)
        if h.GPU:
            input_char = input_char.cuda()

        output, generate_hidden, recurrent_output = model(input_char, generate_hidden)

    output_char = first_chars[-1]
    output_int = h.char2int_cypher[output_char]
    predicted_chars = first_chars
    hidden_activations = []
    i=0
    while True:
        if not till_end and i >= prediction_length:
            # stop at set point, otherwise keep generating till an end
            break
        if till_end and output_char == h.end_sub:
            break
            
        input_char = Variable(torch.LongTensor([output_int])).view(1, -1)
        if h.GPU:
            input_char = input_char.cuda()

        output, generate_hidden, recurrent_output = model(input_char, generate_hidden)
        recurrent_output = recurrent_output.view(-1)
        hidden_activations.append(recurrent_output.cpu().data.numpy())

        #???????
        softmax_dist = torch.nn.functional.softmax(output/temperature)
        output_int = int(torch.multinomial(softmax_dist,1).cpu().data[0].numpy())       

        output_char = h.int2char_cypher[output_int]
        predicted_chars += output_char
        i+=1

    file = open(generate_file, "w")
    file.write(predicted_chars)
    file.close()

    print("final output = ", predicted_chars)
    return predicted_chars[len(first_chars):],np.array(hidden_activations)


def visualize(predicted_chars, hidden_activations, map_width, map_height):
    """
    Function to visualize
    input:
        <str> predicted_chars: the string predicted by generating with our model
        <np.array> hidden_activations: the activations for each predicted character
            (prediction_size x hidden_size)
    """
    special_chars = {'\n':'nl', ' ':'sp'}

    for i in hidden_activations.T: 
        data = i.reshape(map_height, map_width)

        plt.figure(figsize = (10,10))
        heatmap = plt.pcolor(data,cmap='bwr')

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                try:
                    predicted_char = predicted_chars[row*map_width + col]
                except Exception as e:
                    break
                if predicted_char in special_chars: 
                    predicted_char = special_chars[predicted_char]

                plt.text(col + 0.5, row + 0.5, predicted_char,
                    horizontalalignment='center',
                    verticalalignment='center',fontsize = 12)
        plt.colorbar(heatmap)

        plt.show()
    
    
    

