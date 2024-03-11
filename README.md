# RLS_src
clean up code for RLS

## Introduction
To run the training code, specify the input arguments in run_train.sh and run:

<code> sh run_train.sh </code>

To visualize the results from certain checkpoints, specify the input arguments in plot_val.sh and run:

<code> sh plot_val.sh </code>


## TODO
1. check if there is bug of the unified version [done]
2. improve the code by use metaphor [done]
3. print the prediction line and ground truth line with red marks for leg movement. [done]
4. slice more training data by using overlapped ones. [done]
5. change the plot code format accordingly [done]
6. change to BCEWithLogits loss and tune the thresholding.
7. separate left and right leg movements.
