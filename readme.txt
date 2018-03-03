
Code for this project can be found in ./code
Data is is ./data
Loss vs epoch data is found in ./losses


Files inside of ./code

main.py / main.ipynb
Central file for executing functions in our other modules to present the outputs needed for the assignment. 
Neuron visualization function calls can be found in 2 commented lines at the bottom of main.py

my_models.py
Contains functions and classes that are directly related to neural networks and the required outputs for the assignment. 
This file has our RNN model, train function, generate function, and neuron visualization function. 


hyperparameters.py
Various settings and parameters as global variables accessable in all modules so that we can see them all in one place and tweak them easily (note that if using jupyter notebook, you'll need to restart kernel to reflect changes to this file since it caches imports)
Currently contains constants for model hyperparamters, dataset substitutions, pytorch GPU flag and loss function, file paths for saving model state

plotting.py / plotting.ipynb
Contains code to plot loss vs epochs for various training models. The losses are stored as .npy arrays in the ./losses directory. 

utils.py
Contains general tools that do not directly relate to neural networks for preprocessing, doing general checks, etc. 
Has load_music function, permute_list function, create_batch function, replace_startend function, monotonic_increase function, shift_list function, and a print_utils utility class

