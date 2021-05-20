# Paper summary - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

## SGD as the authors found it
Stochastic gradient descent (or _minibatch gradient descent_) iteratively updates weights `theta`, i.e. `theta_{t+1} = theta_t + alpha * del L / del theta_t` for loss `L`, over for a minibatch at a time instead of for a whole dataset at once. Sometimes the term stochastic gradient descent is used to mean minibatch gradient descent, other times it's used to mean gradients are computed one row of the dataset at a time. Using minibatches is thus a midpoint between the two extremes. 

The _saturation problem_ is a property of activation functions. Activation functions which squish are said to be _saturating_ (like sigmoid and tanh), while activation functions which do not squish are said to be _non-saturating_ (like ReLU). The saturation problem implies vanishing gradients. Usually the saturation problem is avoided by using ReLU, initializations such as Glorot, and small learning rates. This paper is proposing we rather than do any of that, just "ensure that the distribution of nonlinearity inputs remains more stable as the network trains". 

### What is _internal covariate shift_ (ICS)
ICS is change in _distribution_ of network activations due to change in network parameters during training. Reducing ICS improves training. 

## Batch normalization's contribution
By fixing the mean and variance of layer inputs, batch normalization
- regularizes the model (reducing the need for dropout)
- reduces the dependence of gradients on the _scale_ of parameters or their initial values
- prevents network from getting stuck in _saturated modes_, making it possible to use _saturating nonlinearities_

## Reducing ICS

You can linearly transform a dataset to have zero mean and unit variance. This procedure is called _standardizing_ or _whitening_. Whitening inputs makes the training process converge faster. We want to go one further, though: we want to add a whitening step between each layer, whiten the activations before sending them to the next matrix, stepping toward a _fixed distribution of inputs_ removing the downsides of ICS. A naive way of whitening does not alert gradient descent to the normalization process. "To address this issue, we would like to ensure that, for any parameter values, the network _always_ produces activations with the desired distribution." In addition, this naive way is expensive. To make matters worse, it's not differentiable. 

Instead, what we do is normalize each feature independently. Note this will not _decorrelate_ the features like the other approach does. We then scale and shift the normalized parameters with two auxiliary learned parameters, so that an identity map is in the space of available functions. Since we are proceeding via minibatch, each minibatch has it's own estimate of the mean and variance we use in the normalization map. This is how normalization participates in backprop! 

## Training vs. inference

We have different needs and wants in training than in inference. In inference, we want the computation to be deterministic, so independent of how batches get carved up. As we saw, it is ok and even desirable for this dependence to be present in training. In inference, we compute normalization with respect to _population_ mean and variance rather than minibatch mean and variance. 

## Batch normalization enables higher learning rates; batch normalization regularizes the model

> Batch normalization makes training more resilient to parameter scale. Normally, large learning rates may increase the scale of layer parameters, which then amplify the gradient during backpropagation and lead to model explosion. However, with batch normalization, backpropagation through a layer is unaffected by the scale of its parameters. 

Another neat property is that batch normalization makes dropout unnecessary. 


## My questions
- I inferred from context that the saturation problem implies vanishing gradients, but I don't see how precisely? 
- what is exactly the problem being talked about when they describe "being stuck in saturated regimes"? 
- Why does reducing ICS improve training? 
- Why is the version of normalizing with the covariance matrix mathematically legitimate? Also, why is it more expensive than computing the denominator on a per-feature basis? 
