# Graded Assignment - Week 1

---

### Q1. [MCQ]  
Suppose a single-layer perceptron is trained on a linearly separable dataset using a step activation. Which of the following guarantees convergence?  

a) Learning rate → 0  
b) Perceptron Convergence Theorem  
c) Universal Approximation Theorem  
d) Vanishing Gradient Theorem  

**Answer:** b  

---

### Q2. [MSQ]  
Consider a neuron with input vector $(x = [x_1, x_2])$, weight vector $(w = [w_1, w_2])$, and bias $(b)$. Let the activation be sigmoid:  

$$
f(z) = \frac{1}{1 + e^{-z}}, \quad z = w_1x_1 + w_2x_2 + b
$$

Which of the following are correct about the **gradient of the output** with respect to $(w_1)$?  

a) $(\frac{\partial f}{\partial w_1} = f(z)(1-f(z))x_1)$  
b) It always lies between -0.25 and 0.25 times $(x_1)$  
c) Gradient vanishes for large $(|z|)$  
d) It is independent of bias $(b)$  

**Answer:** a, b, c  

---

### Q3. [MCQ]  
Which of the following statements about the **Universal Approximation Theorem** is correct?  

a) A feedforward neural network with a single hidden layer and finite neurons can approximate any continuous function on compact subsets of $(\mathbb{R}^n)$.  
b) It guarantees efficient training of neural networks.  
c) It applies only to recurrent neural networks.  
d) It requires infinite hidden layers.  

**Answer:** a  

---

### Q4. [MSQ]  
You design a neural network without any activation functions (only linear transformations). Which statements are true?  

a) The network reduces to a single linear transformation regardless of depth  
b) Such a model cannot represent XOR function  
c) Gradient descent will diverge in training  
d) Adding more layers improves representation power  

**Answer:** a, b  

---

### Q5. [NAT]  
Implement a Python function that computes **binary cross-entropy loss** for given true labels and predicted probabilities. Then compute the loss for:  

$$
y_{true} = [1, 0, 1], \quad y_{pred} = [0.9, 0.2, 0.7]
$$ 

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

y_true = [1, 0, 1]
y_pred = [0.9, 0.2, 0.7]
print(binary_cross_entropy(y_true, y_pred))  
```

**Answer:** ~0.228  

---

### Q6. [MCQ]  
Which activation function can output unbounded positive values, is sparse in nature, and widely used in deep networks?  

a) Sigmoid  
b) ReLU  
c) Tanh  
d) Softmax  

**Answer:** b  

---

### Q7. [MSQ]  
Given ReLU activation function $(f(z) = \max(0, z))$:  

Which of the following are true?  

a) Its derivative is 1 when $(z > 0)$ and 0 when $(z < 0)$  
b) It avoids vanishing gradients in positive domain  
c) It is differentiable everywhere including at $(z=0)$  
d) It induces sparsity in activations  

**Answer:** a, b, d  

---

### Q8. [MCQ]  
You are training a deep network with sigmoid activations in all layers. Training is extremely slow due to vanishing gradients. Which strategy is **least effective**?  

a) Replace sigmoid with ReLU or variants  
b) Normalize inputs and use better weight initialization  
c) Use residual connections or batch normalization  
d) Increase the number of sigmoid layers further  

**Answer:** d  

---

### Q9. [NAT]  
Consider a neuron with inputs $(x = [1,2,3])$, weights $(w = [0.2, -0.5, 0.1])$, bias $(b = 0.4)$, and activation `tanh`. Compute the neuron output (approximate to 3 decimal places).  

$$
f(z) = \tanh(z) 
$$

**Answer:** -0.100  

---

### Q10. [MSQ]  
Which of the following are reasons why loss metrics are essential in neural networks?  

a) They quantify the mismatch between predictions and ground truth  
b) They provide gradients for optimization  
c) They guarantee avoidance of overfitting  
d) They enable comparison between different models  

**Answer:** a, b, d  
