# machineLearning
Image , text processing
What are Channels and Kernels (according to EVA)?
Kernels: A unit. You can think of it as an atom(the smallest particle of a chemical element that can exist).

Channels: Collection of kernels. you can think of it as a molecule(Every combination of atoms is a molecule).


Why should we only (well mostly) use 3x3 Kernels?

a. How to choose between smaller and larger filter size?
b. Why 3x3 and not any other filter like 5x5?

a. How to choose between smaller and larger filter size?

Smaller filter looks at very few pixels at a time hence it has small receptive field. Where as large filter will look at lot of pixel hence it will have large receptive field

When we work with smaller filters, we focus on every minute details and capture smaller complex features from Image, where as when we work with larger filters we tends to search for generic features which will give us basic components.

After capturing smaller/ minute features from Image we can make use of them later in the processing. We loose this benefit with large filters as they focus on generics not specific features.

b. Why 3x3 and not any other filter like 5x5 or 7x7?

Less filter less computation, big filter more computation.

It learns large complex features easily, where as large filters learns simple features.

Output Layers will be less when we use 3x3 filters as compared to 5x5 or bigger filters.

Also since there will be more output layers when using 3x3 filters more memory will be required to store them as compared to 5x5 or bigger filters.

How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)
 199x199 | 3x3 > 197x197
 197x197 | 3x3 > 195x195
 195x195 | 3x3 > 193x193
193x193 | 3x3 > 191x191
191x191 | 3x3 > 189x189
189x189 | 3x3 > 187x187
187x187 | 3x3 > 185x185
185x185 | 3x3 > 183x183
183x183 | 3x3 > 181x181
181x181 | MaxPooling  > 90x90
90x90 | 3x3 > 88x88
88x88 | 3x3 > 86x86
86x86 | 3x3 > 84x84
84x84 | 3x3 > 82x82
82x82 | MaxPooling > 41x41
41x41 | 3x3 > 39x39
39x39 | 3x3 > 37x37
37x37 | 3x3 > 35x35
35x35 | 3x3 > 33x33
33x33 | 3x3 > 31x31
31x31 | MaxPooling > 15x15
15x15 | 3x3 > 13x13
13x13 | 3x3 > 11x11
11x11 | 3x3 > 9x9
9x9  | 3x3 > 7x7
7x7  | 3x3 > 5x5
5x5  | 3x3 > 3x3
3x3  | 3x3 > 1x1


Session3
Number of parameters in keras is different from pytorch. Mention why.


PyTorch doesn't have a function to calculate the total number of parameters as Keras does, but it's possible to sum the number of elements for every parameter group:

pytorch_total_params = sum(p.numel() for p in model.parameters())
If you want to calculate only the trainable parameters:

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

How many layers to use for a given image size 
The Number of convolutional layers: In my experience, the more convolutional layers the better (within reason, as each convolutional layer reduces the number of input features to the fully connected layers), although after about two or three layers the accuracy gain becomes rather small so you need to decide whether your main focus is generalisation accuracy or training time. That said, all image recognition tasks are different so the best method is to simply try incrementing the number of convolutional layers one at a time until you are satisfied by the result.


MaxPooling
Maximum pooling, or max pooling, is a pooling operation that calculates the maximum, or largest, value in each patch of each feature map.
1x1 Convolutions is used to reduce layers.
3x3 Convolutions 
It is used for blurring, sharpening, embossing, edge detection, and more. This is accomplished by doing a convolution between a kernel and an image. ... While applying 2D convolutions like 3X3 convolutions on images, a 3X3 convolution filter, in general will always have a third dimension in size.
Receptive Field 
The receptive field in Convolutional Neural Networks (CNN) is the region of the input space that affects a particular unit of the network. ... The numbers inside the pixels on the left image represent how many times this pixel was part of a convolution step (each sliding step of the filter)
SoftMax
The softmax activation is normally applied to the very last layer in a neural net, instead of using ReLU, sigmoid, tanh, or another activation function. The reason why softmax is useful is because it converts the output of the last layer in your neural network into what is essentially a probability distribution.
Learning Rate
The amount that the weights are updated during training is referred to as the step size or the “learning rate.” Specifically, the learning rate is a configurable hyperparameter used in the training of neural networks that has a small positive value, often in the range between 0.0 and 1.0.
Kernels and how do we decide the number of kernels?
A common choice is to keep the kernel size at 3x3 or 5x5. The first convolutional layer is often kept larger. Its size is less important as there is only one first layer, and it has fewer input channels: 3, 1 by color.
Batch Normalization
Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.
Image Normalization
Subtracting the dataset mean serves to "center" the data. Additionally, you ideally would like to divide by the sttdev of that feature or pixel as well if you want to normalize each feature value to a z-score.

The reason we do both of those things is because in the process of training our network, we're going to be multiplying (weights) and adding to (biases) these initial inputs in order to cause activations that we then backpropogate with the gradients to train the model.

We'd like in this process for each feature to have a similar range so that our gradients don't go out of control (and that we only need one global learning rate multiplier).

Another way you can think about it is deep learning networks traditionally share many parameters - if you didn't scale your inputs in a way that resulted in similarly-ranged feature values (ie: over the whole dataset by subtracting mean) sharing wouldn't happen very easily because to one part of the image weight w is a lot and to another it's too small.

You will see in some CNN models that per-image whitening is used, which is more along the lines of your thinking.
Position of MaxPooling
Maximum pooling, or max pooling, is a pooling operation that calculates the maximum, or largest, value in each patch of each feature map. The results are down sampled or pooled feature maps that highlight the most present feature in the patch, not the average presence of the feature in the case of average pooling.
Concept of Transition Layers
It would be impracticable to concatenate feature maps of different sizes (although some resizing may work). Thus in each dense block, the feature maps of each layer has the same size. However down-sampling is essential to CNN. Transition layers between two dense blocks assure this role.
Position of Transition Layer
Provide input image into convolution layer.
Choose parameters, apply filters with strides, padding if requires. ...
Perform pooling to reduce dimensionality size.
Add as many convolutional layers until satisfied.
Flatten the output and feed into a fully connected layer (FC Layer)
Number of Epochs and when to increase them
You should set the number of epochs as high as possible and terminate training based on the error rates. Just mo be clear, an epoch is one learning cycle where the learner sees the whole training data set. If you have two batches, the learner needs to go through two iterations for one epoch.
DropOut
The term “dropout” refers to dropping out units (both hidden and visible) in a neural network. Simply put, dropout refers to ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random.
When do we introduce DropOut, or when do we know we have some overfitting
You shouldn't do it by looking at the loss or accuracy of training phase. Theoretically, the training accuracy should always be increasing (also means the training loss should always be decreasing) because you train the network to decrease the training loss. But a high training accuracy doesn't necessary mean a high test accuracy, that's what we referred as over-fitting problem. So what you need to find is a point where the accuracy of test set (or validation set if you have it) stops increasing. And you can simply do it by specifying a relatively larger number of iteration at first, then monitor the test accuracy or test loss, if the test accuracy stops increasing (or the loss stops decreasing) in consistently N iterations (or epochs), where N could be 10 or other number specified by you, then stop the training process.
The distance of MaxPooling from Prediction
If block size of 5 is used then max distnce from last max pooling should be 5.
The distance of Batch Normalization from Prediction
When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
Stacking smaller convolutional layers is lighter, than having bigger ones. It also tends to improve the result, with the simple intuition that it results in more layers and deeper networks.
How do we know our network is not going well, comparatively, very early
Batch Size, and effects of batch size
When to add validation checks
LR schedule and concept behind it
Adam vs SGD
SGD is a variant of gradient descent. Instead of performing computations on the whole dataset — which is redundant and inefficient — SGD only computes on a small subset or random selection of data examples. ... Essentially Adam is an algorithm for gradient-based optimization of stochastic objective functions.
