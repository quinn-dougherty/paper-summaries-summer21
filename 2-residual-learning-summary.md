# Summary of "Residual Learning for Image Recognition"

A mysterious problem in deep learning, consider only convolutional classifiers for computer vision applications, is _degradation_ or the loss of accuracy from the increase in model depth. Degradation is mysterious because it is not a matter of overfitting; it is a problem on the train set not just the test set. The authors propose adding an addition circuit to groups of layers and wrapping an activation around a group of layers to plug into that circuit, effectively adding an identity function to the next activation. Concretely, if `F(x)` is of the form `W(sigma(V(x)))` for nonlinearity `sigma` and weights matrices `W` and `V`, we are interested in `H(x) = F(x) + x`. This is called a _shortcut connection_. 

Here, `H(x)` is the underlying map that the learning process is searching for, and since `F(x)` looks like `H(x) - x`, we call it a _residual_. 

We can visualize shortcut connections in pytorch like so
```python
class ResNet(nn.Module): 
    """6 layers with two shortcut connections """
    ...
    def forward(self, x): 
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out) + x
        out2 = self.conv3(out)
        out2 = self.relu(out2)
        out2 = self.conv4(out2)
        out2 = self.relu(out2) + out
        out3 = self.fc1(out2)
        out3 = self.relu(out3)
        out3 = self.fc2(out3)
        return out3
```

The results of using shortcut connections is deeper models get more accurate at deeper network sizes than smaller network sizes. The control architecture ("plain") performs worse with more layers, because of the degradation problem, but the paper's contribution architecture ("ResNet") dodges the degradation problem and improves performance. 

## My questions
> Identity shortcut connections add neither extra parameter nor computational complexity.
1. is parameter complexity just number of parameters? By computational cmplexity do they mean space,Btime, or both? 
2. Shortcut connections do not increase the number of FLOPs. How/why? 
