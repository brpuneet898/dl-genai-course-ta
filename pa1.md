# Practice Assignment - 1 (Week 1)

---

### Q1. [MCQ]  
Which of the following best describes an *artificial neuron* in the context of neural networks?  

a) A function that maps discrete values to binary states  
b) A computational unit that applies a weighted sum of inputs followed by an activation function  
c) A mathematical model that only computes derivatives  
d) A data structure used for storing neural parameters  

**Answer:** b  

---

### Q2. [MSQ]  
Which of the following are **roles of activation functions** in a neural network?  

a) Introduce non-linearity into the model  
b) Prevent weights from updating during training  
c) Allow the network to approximate complex functions  
d) Help gradient flow during backpropagation  

**Answer:** a, c, d  

---

### Q3. [MCQ]  
Given an artificial neuron with inputs $(x_1 = 2, x_2 = -1)$, weights $(w_1 = 0.5, w_2 = -0.25)$, and bias $(b = 1)$, compute the weighted sum $(z)$.  

$$
z = w_1x_1 + w_2x_2 + b
$$

a) 0.25  
b) 1.75  
c) 2.25  
d) -0.75  

**Answer:** c  

---

### Q4. [NAT]  
Implement a **Python function** that computes the output of a neuron given input vector $(x)$, weights $(w)$, bias $(b)$, and activation function $(\sigma)$. Test it with ReLU activation.  

```python
import numpy as np

def relu(z):
    return max(0, z)

def neuron_output(x, w, b, activation):
    z = np.dot(w, x) + b
    return activation(z)

x = np.array([2, -1])
w = np.array([0.5, -0.25])
b = 1
print(neuron_output(x, w, b, relu))
```

**Answer:** `2.25` 

---

### Q5. [MCQ]  
Which of the following is **NOT a commonly used activation function**?  

a) Sigmoid  
b) Tanh  
c) ReLU  
d) Euclidean  

**Answer:** d  

---

### Q6. [MSQ]  
Consider the **sigmoid activation function**:  

$$ 
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Which of the following statements are true?  

a) Its output always lies between 0 and 1  
b) It suffers from the vanishing gradient problem for large $(|z|)$  
c) It is zero-centered  
d) It is differentiable everywhere  

**Answer:** a, b, d  

---

### Q7. [NAT]  
Write a Python function that computes **Mean Squared Error (MSE)** given predicted values $(y_{pred})$ and true values $(y_{true})$. Test it with:  
$$
y_{true} = [1, 0, 1], \quad y_{pred} = [0.9, 0.2, 0.8]
$$  

```python
import numpy as np

def mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred))**2)

y_true = [1, 0, 1]
y_pred = [0.9, 0.2, 0.8]
print(mse(y_true, y_pred)) 
```

**Answer:** 0.0233 (approx).  

---

### Q8. [MCQ]  
Which loss function is most suitable for **binary classification** problems?  

a) Mean Squared Error (MSE)  
b) Cross-Entropy Loss  
c) Hinge Loss  
d) KL Divergence  

**Answer:** b  

---

### Q9. [MSQ]  
Why are **non-linear activation functions** essential in deep networks?  

a) Without them, multiple layers collapse into a single linear transformation  
b) They allow networks to approximate complex non-linear decision boundaries  
c) They reduce computational cost compared to linear functions  
d) They enable learning of high-dimensional representations  

**Answer:** a, b, d  

---

### Q10. [MSQ]  
Which of the following challenges can occur when training deep neural networks?  

a) Vanishing or exploding gradients  
b) Overfitting due to high model capacity  
c) Difficulty in optimization due to non-convexity  
d) Infinite solutions due to regularization  

**Answer:** a, b, c  
