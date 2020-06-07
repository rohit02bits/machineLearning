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

