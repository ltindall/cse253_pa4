import numpy as np
import hyperparameters as h

def load_music(file_name, use_custom=False):
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

    if use_custom:
        content = replace_startend(content, h.start_sub, h.end_sub)
    
    unique = sorted(set(content))
    char_to_int = {}
    int_to_char = {}
    for i,ch in enumerate(unique):
        char_to_int[ch] = i
        int_to_char[i] = ch

    converted = [char_to_int[ch] for ch in content]

    return converted, char_to_int, int_to_char


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
        <np.array> sequence_size x batch_size from the inputs
        <np.array> the target sequences
        None if not enough data for a batch
    """
    batch = []
    targets = []
    start_idxs = []
    for i in range(batch_size):
        if permuted_list:
            start_idx = permuted_list.pop()
            sequence = inputs[start_idx:start_idx+sequence_size]
            target = inputs[start_idx+1:start_idx+sequence_size+1]
            batch.append(sequence)
            targets.append(target)
            start_idxs.append(start_idx) # don't really need this, just for debugging
        else:
            return None, None, None
    return np.array(batch).T, np.array(targets).T, start_idxs


def replace_startend(input_str, start, end):
    """
    Replaces the <start> and <end> tags with special chars 
    
    input:
        <str> input_str: the input string to parse and replace
        <char> start: the special char to replace <start>
        <char> end: the special char to replace <end>
    return:
        <str> the updated string
    """
    return input_str.replace('<start>', start).replace('<end>', end)
    