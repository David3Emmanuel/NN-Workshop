# Neural Networks Workshop - Instructor Notes

## Workshop Preparation

- Verify all participants have the required Python libraries installed
- Have backup solutions ready for common installation issues
- Pre-download the MNIST dataset if internet connection is unreliable
- Test the notebook on your system before the workshop

## Teaching Tips

### Introduction (20 min)

- Use visual aids when explaining neural network concepts
- Draw a simple neural network on a whiteboard/slide
- Relate neural networks to biological neurons briefly
- Emphasize core concepts: weights, biases, activation functions, cost functions

### Setup (15 min)

- Help troubleshoot installation issues
- Explain each library's purpose
- Monitor participants to ensure everyone is set up

### Building Neural Networks (40 min)

- From Scratch Implementation (Primary Focus):
  - Walk through the mathematics of forward propagation
  - Show how to implement activation functions (sigmoid, ReLU)
  - Demonstrate backpropagation math and code step by step
  - Implement simple weight updates with gradient descent
  - Code a complete neural network class from scratch
- PyTorch Implementation:
  - Explain each step of the PyTorch code
  - Compare with the from-scratch approach to highlight abstractions
  - Highlight the relationship between code and concepts
  - Explain why we normalize data (0-1 range)

### Experimentation (30 min)

- Suggest specific parameters to modify
- Ask participants to predict outcomes before running
- Have participants share interesting results
- Discuss patterns in what improves/hurts performance

### Advanced Concepts (10 min)

- Connect to what they've just done
- Show real-world examples if possible
- Keep explanations conceptual without diving into complex math

### Q&A (5 min)

- Be prepared for common questions:
  - "How do I choose the number of layers/neurons?"
  - "What's the difference between various activation functions?"
  - "How does this apply to real-world problems?"
  - "What hardware is needed for bigger networks?"

## Common Challenges

- PyTorch installation issues
- Confusion about optimizer and loss function selection
- Understanding hyperparameters vs. learned parameters
- Interpreting accuracy/loss curves
- Common errors in from-scratch implementation

## Extended Activities (if time permits)

- Transfer learning example
- Image classification with a pre-trained model
- Saving and loading models
- Visualizing layer activations

## Follow-up Resources

Prepare a list of beginner-friendly resources for continued learning
