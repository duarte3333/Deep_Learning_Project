## Homework 1

### Question 1: Medical Image Classification with Linear Classifiers and Neural Networks (35 points)
- **Task 1.1:** Implement a perceptron for OCTMNIST dataset classification.
  - **Requirements:** Train for 20 epochs, report performance, plot accuracies.
- **Task 1.2:** Use logistic regression with stochastic gradient descent.
  - **Requirements:** Compare two learning rates (η = 0.01, η = 0.001), report accuracies.
- **Task 1.3:** Implement a multi-layer perceptron (MLP) with a single hidden layer.
  - **Requirements:** Use 200 hidden units, relu activation, and cross-entropy loss. Train for 20 epochs.

### Question 2: Medical Image Classification with an Autodiff Toolkit (35 points)
- **Task 2.1:** Implement a linear model with logistic regression using a deep learning framework.
  - **Requirements:** Train for 20 epochs, optimize learning rate, report test accuracy.
- **Task 2.2:** Implement a feed-forward neural network with dropout regularization.
  - **Requirements:** Compare performance with different batch sizes and learning rates, assess overfitting, experiment with L2 regularization and dropout.

### Question 3: Boolean Function Computation with Multilayer Perceptron (30 points)
- **Task 3.1:** Demonstrate the insufficiency of a single perceptron for a specific Boolean function.
- **Task 3.2:** Show how a multilayer perceptron with a single hidden layer can compute the function.
- **Task 3.3:** Solve the same problem using rectified linear unit activations.
  - **Points:** 10

## Homework 2

### Question 1: Transformer Self-Attention Layer (30 points)
- **Task 1.1:** Self-Attention Complexity 
  - **Requirements:** Calculate the computational complexity of the self-attention layer for sequence length L and hidden size D. Discuss why this is problematic for long sequences.
- **Task 1.2:** Softmax Approximation 
  - **Requirements:** Approximate softmax using McLaurin series expansion. Define a feature map φ, express M as a function of D, and discuss dimensionality for K ≥ 3 terms.
- **Task 1.3:** Self-Attention Operation Approximation 
  - **Requirements:** Approximate the self-attention operation using φ and discuss its computational efficiency.
- **Task 1.4:** Complexity Reduction 
  - **Requirements:** Show how the approximation leads to linear complexity in L and discuss its dependency on M and D.

### Question 2: Image Classification with CNNs (35 points)
- **Task 2.1:** Implement a Simple Convolutional Network 
  - **Requirements:** Use the OCTMNIST dataset. Design the network with specific layers and train using SGD. Tune the learning rate and report findings.
- **Task 2.2:** Network without Max-Pooling Layers 
  - **Requirements:** Implement a similar network without max-pooling. Report performance using optimal hyperparameters.
- **Task 2.3:** Trainable Parameters 
  - **Requirements:** Implement `get_number_trainable_params` and discuss performance differences between networks.

### Question 3: Automatic Speech Recognition (35 points)
- **Task 3.1:** Recurrent-based Decoder Implementation 
  - **Requirements:** Implement the `_forward` method of `TextDecoderRecurrent`. Compare training/validation loss and report test loss and string similarity scores.
- **Task 3.2:** Attention-based Decoder Implementation 
  - **Requirements:** Implement the `forward` method of `TextDecoderTramsformer`. Perform similar comparisons and reporting as in part 1.
- **Task 3.3:** Test Results Comparison 
  - **Requirements:** Discuss differences in test results between the two decoder architectures.
- **Task 3.4:** String Similarity Score Analysis 
  - **Requirements:** Analyze and comment on the different string similarity scores used.
