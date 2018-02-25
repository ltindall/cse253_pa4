import numpy as np

def load_music(file_name):
    """
    loads the music file so that it becomes a list of ints
    enumerated based on ascii order
    
    input: filename <string>
    return: 
        <list> of ints that represent the input file
        <dict> for character to int conversion
    """
    # load the file    
    with open('../data/{}'.format(file_name), 'r') as file:
        content = file.read()

    unique = sorted(set(content))

    char_to_int = {}
    for i,ch in enumerate(unique):
        char_to_int[ch] = i

    converted = [char_to_int[ch] for ch in content]

    return converted, char_to_int


def permute_list(num_chars, sequence_size, overlap=True):
    """
    Creates a random permutation of the input list to use in an epoch
    
    inputs:
        <int> num_chars: the number of characters in the data text
        <int> sequence_size: the size of the chunks of the sequence
        <boolean> overlap: whether we want the sequences to overlap
    return:    
        <list> list of numbers in a random sequence permutation
    """
    if overlap:
        valid_starts = num_chars - sequence_size
    else:
        last_valid = num_chars - sequence_size
        valid_starts = np.arange(0,last_valid,sequence_size-1)
        
    return list(np.random.permutation(valid_starts))


def create_batch(inputs, permuted_list, batch_size, sequence_size):
    """
    Creates a batch from the inputs based on the given permute array
    
    input:
        <list> inputs: the input list of integers
        <list> permuted_list: the permutation of valid start indices in the input_list
        <int> batch_size: the batch size
        <int> sequence_size: the size of the sequence
    return:
        <np.array> batch_size x sequence_size from the inputs
        None if not enough data for a batch
    """
    batch = []
    start_idxs = []
    for i in range(batch_size):
        if permuted_list:
            start_idx = permuted_list.pop()
            sequence = inputs[start_idx:start_idx+sequence_size]
            batch.append(sequence)
            start_idxs.append(start_idx)
        else:
            return None
    return np.array(batch), start_idxs
