# CSE 253 PA4 




good blog on rnn's: http://karpathy.github.io/2015/05/21/rnn-effectiveness/  
good blog on lstm's: http://colah.github.io/posts/2015-08-Understanding-LSTMs/  


useful code repos:  
https://github.com/spro/practical-pytorch/tree/master/char-rnn-generation  
The repo above has some problems. One epoch only trains over a single random 
sequence from the text. It doesn't use batches.  

https://gist.github.com/kylemcdonald/0518aa9e63e2514073fbf6efd506be20  
This repo does epochs and batches correctly. It splits the text into sequences
of 50 chars and then some number of batches of 50 chars each. 


## Summary:
### Files (classes and function names are bolded)
  - my_models.py 
    - contains functions and classes that are directly related to neural networks and the required outputs for the assignment
    - has our **lstm model**, **train** function, **generate** function
    
  - utils.py
    - contains general tools that do not directly relate to neural networks for preprocessing, doing general checks, etc
    - has **load_music** function, **permute_list** function, **create_batch** function, **replace_startend** function, **monotonic_increase** function, **shift_list** function, and a **print_utils** utility class
  
  - hyperparameters.py
    - various settings and parameters as global variables accessable in all modules so that we can see them all in one place and tweak them easily (note that if using jupyter notebook, you'll need to restart kernel to reflect changes to this file since it caches imports)
    - currently contains constants for **model hyperparamters**, **dataset substitutions**, **pytorch GPU flag and loss function**, **file paths for saving model state**
    
   - main.py / main.ipynb
     - central file for executing functions in our other modules to present the outputs needed for the assignment
  
