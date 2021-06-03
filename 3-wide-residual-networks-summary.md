# Wide Residual Networks - Summary

We saw [last time](https://github.com/quinn-dougherty/paper-summaries-summer21/blob/main/2-residual-learning-summary.md) that _residual blocks_ or _shortcut connections_ increase model performance and decrease training time. Today, we will see that this work can be extended in the direction of "width beats depth". 

Network _width_ is better thought of as "height" in the traditional left-to-right graphical view. It is the number of neurons in a hidden layer. network _depth_ is better thought of as "width" in the traditional left-to-right graphical view. It is the number of hidden layers. For residual blocks in CNNs, the width of a block is the number of channels (??, uncertain). In today's paper, the authors show that a network only 16 layers deep can outperform a network of a thousand layers. 

The three basic ways to increase representational power of residual blocks is to 1. add more convolutional layers per block, 2. widen the convolutional layers by adding more feature planes, and 3. increase filter sizes in the convolutional layers. The paper is mostly about 2., and pretty much takes 1. and 3. as fixed. One hint that the authors point to is that pre-ResNet architectures often favored wider choices, so it would have been natural to revisit that in a post-ResNet context. 

Like before, a residual block is just an additive circuit combining an identity function and a function of the form `x -> W(sigma(V(x)))` for nonlinearity `sigma` and weights matrices `W` and `V`. A residual network is just a sequential stack of residual blocks. In this paper, the authors use a _deepening factor_ `l` and a _widening factor_ `k`, where `l` is the number of convolutions in a block and `k` multiplies the number of features in convolutional layers. 

Over the course of some experiments, the authors found that 
- higher values of `k` imply better performance
- increasing both `l` and `k` are good until some sort of threshold after which regularization is needed
- high depth doesn't entail regularization effect
> there doesn’t seem to be a regularization effect from very high depth in residual networks as wide networks with the same number of parameters as thin ones can learn same or better representations. Furthermore, wide networks can successfully learn with a 2 or more times larger number of parameters than thin ones, which would require doubling the depth of thin networks, making them infeasibly expensive to train.

The impact of dropout is marginal in this setting. 
