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

