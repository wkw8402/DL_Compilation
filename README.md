# Deep Learning

<aside>
💡

</aside>

### Date: May 5, 2019

### Title: Deep Learning Intro

- Deep learning is machine learning using neural network.
- Deep learning does not require domain knowledge.
- Deep learning is good at image classification/detection, sequence data analysis/prediction.
- Machine Learning types:
    - supervised learning - data has correct answers
    - unsupervised learning - want computer to classify by itself.
    - reinforcement learning - teaching how to play game

$$
D_1 * weight_1\;+\;D_2 * weight_2 + bias
$$

- Tell computer to compute weights that minimize the difference.
- **Perceptron** = neural network with connected layers
- Feature extraction in deep learning is done by computer without any prior knowledge
- In each (hidden) layer, we have node(s) that are numbers which is equal to sum of all previous nodes multiplied by the corresponding connection weights.
- Loss function evaluates the accuracy of the model and computes the difference of the model.
- To predict integers, we use Mean Squared Error:

$$
\frac{1}{n} \Sigma (\hat{y}-y)^2
$$

- To predict probability, we use Binary Cross Entropy:

$$
-\frac{1}{n} \Sigma [y\;log(\hat{y}) + (1-y)log(1-\hat{y})]
$$

- Activation function changes the value in the node to make non-linear complex prediction.
    
    $$
    sigmoid = \frac{1}{1 + e^{-x}}
    $$
    
    $$
    hyperbolic\;tangent = \frac{e^x - e^{-x}}{e^x + e^{-x}}
    $$
    
    $$
    rectified\;linear\;units = x\;if\;x > 0,\;0\;otherwise
    $$
    
- Gradient descent = subtract the gradient of tangent from current weight to find optimal weight and minimize total loss.

$$
w_1 \leq w_1 - \alpha \frac{\delta E}{\delta w_1}
$$

- **gradient** (of tangent in 2d) = how much the weight change affect the total loss.
- Deep learning process:
    1. randomly choose weights
    2. calculate total loss based on weights
    3. use gradient descent to update weights to new weights
    4. repeat 1-3 until total loss does not decrease
- Learning rate, $\alpha$, is multiplied to gradient to avoid falling into local minima and find the true minimum.
- Learning rate optimizer enables quicker finding of the true minimum.
    - Momentum: maintain the acceleration
    - AdaGrad: small learning rate for frequently changing w, big learning rate for not changing w
    - RMSProp: squared AdaGrad
    - AdaDelta: AdaGrad but prevent learning rate from being too small and have negligible effect
    - **Adam: RMSProp + Momentum**
- Back Propagation

![Untitled](Deep%20Learning%20fc49624085554401ab6c91f81f2aa831/Untitled.png)

- Update weights closer last layer first.
- $z_1 = input * w_1, a_1 = sigmoid(z_1)$

$$
w_3 \leq w_3 - \alpha \frac{\delta E}{\delta w_3}
$$

$$
\frac{\delta E}{\delta w_3} = \frac{\delta z_3}{\delta w_3} * \frac{\delta a_3}{\delta z_3} * \frac{\delta E}{\delta a_3}
$$

$$
z_3 = w_3a_1 + w_4a_2
$$

$$
\frac{\delta z_3}{\delta w_3} = a1
$$

$$
a_3 = \frac{1}{1 + e^{-z_3}}
$$

$$
\frac{\delta a_3}{\delta z_3} =  sig(z_3)*(1-sig(z_3)) 
$$

$$
E = \frac{1}{2}((a_3 - y_1)^2+(a_4-y_2)^2 = \frac{1}{2}({a_3}^2-2a_3 y_1 + {y_1}^2 + ...)
$$

$$
\frac{\delta E}{\delta a_3} = \frac{1}{2}(2a_3 - 2y_1) = a_3 - y_1
$$

### Convolution layer:

Feature extraction: make 20 duplicates, each having image's different important features
deep learning based on feature extraction
used to apply kernels(ex. sharpen, gaussian blur)=filter to make layer

goal1 = increase val_accuracy by adding dense layer? adding conv+pooling?
goal2 = prevent overfitting by stop learning when val_accuracy stops increasing

**overfitting** = memorize training dataset, happens when last epoch accuracy is greater than evaluation

Convolutional Neural Network:
Input image -> Filters -> Convolutional layer -> pooling -> Flattening -> Dense -> Output

pooling layer(down sampling):
max pooling summarize areas with max values

<aside>
📌 Summary**:**

</aside>
