## What was learned?
The tasks across both homework assignments require proficiency in implementing various machine learning and deep learning algorithms, including perceptrons, logistic regression, multi-layer perceptrons, convolutional neural networks, and transformers, using both low-level libraries like NumPy and high-level frameworks such as PyTorch. The project was done by Me and @AlexReis1313 .

## Homework 1

### Question 1: Medical Image Classification with Linear Classifiers and Neural Networks (35 points)

<p align="center">
  <img src="https://github.com/duarte3333/Deep_Learning_Project/assets/76222459/186963b7-2a41-4c58-b925-a5424a4546c3" alt="image"/>
</p>

- **Task 1.1:** Implement a perceptron for OCTMNIST dataset classification with only numpy.
  - **Requirements:** Train for 20 epochs, report performance, plot accuracies.
- **Task 1.2:** Implement logistic regression with stochastic gradient descent with only numpy
  - **Requirements:** Compare two learning rates (η = 0.01, η = 0.001), report accuracies.
- **Task 1.3:** Implement a multi-layer perceptron (MLP) with a single hidden layer using numpy only.
  - **Requirements:** Use 200 hidden units, relu activation, and cross-entropy loss. Train for 20 epochs.

### Question 2: Medical Image Classification with an Autodiff Toolkit (35 points)

<p align="center">
  <img src="https://github.com/duarte3333/Deep_Learning_Project/assets/76222459/e04d53ee-4687-45f9-b01a-cf6ed5d22bc2" alt="image", width=60% />
</p>

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

<p align="center">
  <img src="https://github.com/duarte3333/Deep_Learning_Project/assets/76222459/a5380161-5cf9-4549-a0bb-f8f68157f0b2" alt="image", width=60% />
</p>

- **Task 1.1:** Self-Attention Complexity 
  - **Requirements:** Calculate the computational complexity of the self-attention layer for sequence length L and hidden size D. Discuss why this is problematic for long sequences.
- **Task 1.2:** Softmax Approximation 
  - **Requirements:** Approximate softmax using McLaurin series expansion. Define a feature map φ, express M as a function of D, and discuss dimensionality for K ≥ 3 terms.
- **Task 1.3:** Self-Attention Operation Approximation 
  - **Requirements:** Approximate the self-attention operation using φ and discuss its computational efficiency.
- **Task 1.4:** Complexity Reduction 
  - **Requirements:** Show how the approximation leads to linear complexity in L and discuss its dependency on M and D.

### Question 2: Image Classification with CNNs (35 points)

<p align="center">
  <img src="https://github.com/duarte3333/Deep_Learning_Project/assets/76222459/020c2664-0430-4c08-8acb-d502b8643e24" alt="image", width=60% />
</p>

- **Task 2.1:** Implement a Simple Convolutional Network 
  - **Requirements:** Use the OCTMNIST dataset. Design the network with specific layers and train using SGD. Tune the learning rate and report findings.
- **Task 2.2:** Network without Max-Pooling Layers 
  - **Requirements:** Implement a similar network without max-pooling. Report performance using optimal hyperparameters.
- **Task 2.3:** Trainable Parameters 
  - **Requirements:** Implement `get_number_trainable_params` and discuss performance differences between networks.

### Question 3: Automatic Speech Recognition (35 points)

<p align="center">
  <img src="https://github.com/duarte3333/Deep_Learning_Project/assets/76222459/4bebdf36-bcac-4b57-b62e-57044f9c8149" alt="image", width=60% />
</p>

- **Task 3.1:** Recurrent-based Decoder Implementation 
  - **Requirements:** Implement the `_forward` method of `TextDecoderRecurrent`. Compare training/validation loss and report test loss and string similarity scores.
- **Task 3.2:** Attention-based Decoder Implementation 
  - **Requirements:** Implement the `forward` method of `TextDecoderTramsformer`. Perform similar comparisons and reporting as in part 1.
- **Task 3.3:** Test Results Comparison 
  - **Requirements:** Discuss differences in test results between the two decoder architectures.
- **Task 3.4:** String Similarity Score Analysis 
  - **Requirements:** Analyze and comment on the different string similarity scores used.
