"""
Neural Networks Workshop Consolidated Script
-------------------------------------------
This script combines the key components from the Neural Networks Workshop series.
It includes:
1. Introduction and implementation of a neural network from scratch
2. Working with the MNIST dataset
3. Implementation using PyTorch
4. Comparison between custom and PyTorch implementations

Author: Neural Networks Workshop
Date: June 14, 2025
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# For reproducibility
np.random.seed(42)

# Check for PyTorch installation and import if available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import datasets, transforms
    
    # Set PyTorch seed for reproducibility
    torch.manual_seed(42)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PYTORCH_AVAILABLE = True
    print(f"PyTorch is available. Using device: {device}")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch is not installed. Only NumPy implementations will be available.")

###########################################
# Part 1: Introduction and Activation Functions
###########################################

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    return x * (1 - x)

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU function."""
    return np.where(x > 0, 1, 0)

def tanh(x):
    """Hyperbolic tangent activation function."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh function."""
    return 1 - np.square(np.tanh(x))

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of leaky ReLU function."""
    return np.where(x > 0, 1, alpha)

def softmax(x):
    """Softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def plot_activation_functions():
    """Plot common activation functions for visualization."""
    x = np.linspace(-5, 5, 100)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x, sigmoid(x))
    plt.title('Sigmoid')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x, relu(x))
    plt.title('ReLU')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x, tanh(x))
    plt.title('Tanh')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x, leaky_relu(x))
    plt.title('Leaky ReLU')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

###########################################
# Part 2: Neural Network From Scratch
###########################################

class NeuralNetworkFromScratch:
    """
    A simple neural network implementation from scratch using NumPy.
    
    This implementation includes:
    - A single hidden layer
    - Sigmoid activation for hidden layer and softmax for output
    - Forward and backward propagation
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        # Backpropagation
        # Convert y to one-hot encoding
        y_one_hot = np.zeros((y.size, output.shape[1]))
        y_one_hot[np.arange(y.size), y] = 1
        
        # Calculate gradients
        dz2 = output - y_one_hot
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def train(self, X, y, learning_rate=0.1, epochs=1000):
        # Training loop
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss
            y_one_hot = np.zeros((y.size, output.shape[1]))
            y_one_hot[np.arange(y.size), y] = 1
            loss = -np.sum(y_one_hot * np.log(output + 1e-8)) / y.size
            losses.append(loss)
            
            # Backpropagation
            dW1, db1, dW2, db2 = self.backward(X, y, output)
            
            # Update weights and biases
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        # Make predictions
        output = self.forward(X)
        return np.argmax(output, axis=1)

class MultiLayerNeuralNetwork:
    """
    A neural network with multiple hidden layers implemented from scratch using NumPy.
    
    Features:
    - Configurable number of layers and neurons
    - ReLU activation for hidden layers, softmax for output
    - Support for mini-batch gradient descent
    """
    def __init__(self, layer_sizes):
        """Initialize a neural network with multiple layers.
        
        Args:
            layer_sizes: List of integers, representing the size of each layer
                         (including input and output layers)
        """
        self.num_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            # Xavier initialization for weights
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 
                               np.sqrt(1 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
    def forward(self, X):
        """Forward propagation."""
        self.Z = []
        self.A = [X]  # Input as the first activation
        
        # Hidden layers with ReLU activation
        for i in range(self.num_layers - 1):
            z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            a = relu(z)
            self.Z.append(z)
            self.A.append(a)
        
        # Output layer with softmax activation
        z = np.dot(self.A[-1], self.weights[-1]) + self.biases[-1]
        a = softmax(z)
        self.Z.append(z)
        self.A.append(a)
        
        return self.A[-1]
    
    def backward(self, X, y, output):
        """Backward propagation."""
        batch_size = X.shape[0]
        gradients = {'dW': [], 'db': []}
        
        # Convert y to one-hot encoding
        y_one_hot = np.zeros((y.size, output.shape[1]))
        y_one_hot[np.arange(y.size), y] = 1
        
        # Output layer error
        dZ = output - y_one_hot
        
        for layer in range(self.num_layers - 1, -1, -1):
            # Calculate gradients for this layer
            dW = (1/batch_size) * np.dot(self.A[layer].T, dZ)
            db = (1/batch_size) * np.sum(dZ, axis=0, keepdims=True)
            
            # Store gradients
            gradients['dW'].insert(0, dW)
            gradients['db'].insert(0, db)
            
            if layer > 0:
                # Compute error for previous layer
                dA = np.dot(dZ, self.weights[layer].T)
                dZ = dA * relu_derivative(self.A[layer])
        
        return gradients
    
    def train(self, X, y, learning_rate=0.01, epochs=1000, batch_size=32, print_every=100):
        """Training the neural network."""
        n_samples = X.shape[0]
        n_batches = max(n_samples // batch_size, 1)
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            epoch_correct = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Calculate loss
                y_one_hot = np.zeros((y_batch.size, output.shape[1]))
                y_one_hot[np.arange(y_batch.size), y_batch] = 1
                batch_loss = -np.sum(y_one_hot * np.log(output + 1e-8)) / y_batch.size
                epoch_loss += batch_loss * (end_idx - start_idx)
                
                # Calculate accuracy
                predictions = np.argmax(output, axis=1)
                epoch_correct += np.sum(predictions == y_batch)
                
                # Backward pass
                gradients = self.backward(X_batch, y_batch, output)
                
                # Update weights and biases
                for layer in range(self.num_layers):
                    self.weights[layer] -= learning_rate * gradients['dW'][layer]
                    self.biases[layer] -= learning_rate * gradients['db'][layer]
            
            # Calculate epoch loss and accuracy
            epoch_loss /= n_samples
            epoch_accuracy = epoch_correct / n_samples * 100
            
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        return losses, accuracies
    
    def predict(self, X):
        """Make predictions."""
        output = self.forward(X)
        return np.argmax(output, axis=1)

def test_xor_problem():
    """Test the neural network on the XOR problem."""
    print("\nTesting neural network on XOR problem...")
    # Create XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    # Train simple network with one hidden layer
    print("Training simple network...")
    nn_scratch = NeuralNetworkFromScratch(input_size=2, hidden_size=4, output_size=2)
    losses = nn_scratch.train(X, y, learning_rate=0.5, epochs=5000)
    
    # Test the network
    predictions = nn_scratch.predict(X)
    print("Simple Network Results:")
    print("Predictions:", predictions)
    print("Actual:     ", y)
    print(f"Accuracy: {np.mean(predictions == y) * 100:.2f}%")
    
    # Train multi-layer network
    print("\nTraining multi-layer network...")
    layer_sizes = [2, 8, 4, 2]  # Input, hidden layers, output
    mlnn = MultiLayerNeuralNetwork(layer_sizes)
    losses, accuracies = mlnn.train(X, y, learning_rate=0.05, epochs=2000, 
                                  batch_size=4, print_every=500)
    
    # Test multi-layer network
    predictions = mlnn.predict(X)
    print("Multi-layer Network Results:")
    print("Predictions:", predictions)
    print("Actual:     ", y)
    print(f"Accuracy: {np.mean(predictions == y) * 100:.2f}%")

    # Create decision boundary visualization
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh grid points
    Z = mlnn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary for XOR Problem (Multi-Layer Network)')
    plt.show()

###########################################
# Part 3: MNIST Dataset Functions
###########################################

def load_mnist_data():
    """
    Load MNIST dataset using PyTorch's torchvision.
    Returns training and testing data loaders.
    """
    if not PYTORCH_AVAILABLE:
        print("PyTorch is not available. Cannot load MNIST dataset.")
        return None, None
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the MNIST dataset
    try:
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        # Create data loaders
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        return train_loader, test_loader
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}")
        return None, None

def get_mnist_subset(loader, num_samples):
    """
    Extract a subset of MNIST for faster training with custom model.
    
    Args:
        loader: DataLoader for MNIST
        num_samples: Number of samples to extract
    
    Returns:
        X, y: Flattened images and labels
    """
    if not PYTORCH_AVAILABLE:
        print("PyTorch is not available. Cannot create MNIST subset.")
        return None, None
        
    samples = []
    labels = []
    samples_count = 0
    
    for images, batch_labels in loader:
        # Flatten the images
        batch_size = images.size(0)
        flattened_images = images.view(batch_size, -1).numpy()
        batch_labels_np = batch_labels.numpy()
        
        # Add to our collection
        samples.append(flattened_images)
        labels.append(batch_labels_np)
        
        # Update count
        samples_count += batch_size
        if samples_count >= num_samples:
            break
    
    # Combine all batches
    X = np.vstack(samples)[:num_samples]
    y = np.concatenate(labels)[:num_samples]
    
    return X, y

def create_balanced_mini_dataset(loader, samples_per_class=50):
    """
    Create a balanced mini-dataset with equal number of samples per class.
    
    Args:
        loader: DataLoader for MNIST
        samples_per_class: Number of samples to include for each digit
    
    Returns:
        X, y: Flattened images and labels
    """
    if not PYTORCH_AVAILABLE:
        print("PyTorch is not available. Cannot create balanced dataset.")
        return None, None
        
    X = []
    y = []
    counts = np.zeros(10, dtype=int)
    
    for images, labels in loader:
        batch_size = images.size(0)
        flattened_images = images.view(batch_size, -1).numpy()
        
        for i in range(batch_size):
            label = labels[i].item()
            if counts[label] < samples_per_class:
                X.append(flattened_images[i])
                y.append(label)
                counts[label] += 1
        
        if np.min(counts) >= samples_per_class:
            break
    
    return np.array(X), np.array(y)

def explore_mnist_dataset():
    """Explore and visualize the MNIST dataset."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch is not available. Cannot explore MNIST dataset.")
        return
        
    train_loader, test_loader = load_mnist_data()
    if train_loader is None:
        return
        
    # Get a batch of training data
    examples = iter(train_loader)
    images, labels = next(examples)
    
    # Print shape information
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample image shape: {images[0].shape}")
    
    # Display a few examples
    plt.figure(figsize=(12, 8))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Create and save a subset for training custom model
    print("\nCreating MNIST subsets for custom model training...")
    X_train_subset, y_train_subset = get_mnist_subset(train_loader, 5000)
    X_test_subset, y_test_subset = get_mnist_subset(test_loader, 1000)
    
    print(f"Training subset shape: {X_train_subset.shape}")
    print(f"Test subset shape: {X_test_subset.shape}")
    
    # Create balanced mini-dataset
    print("\nCreating balanced mini-dataset...")
    X_mini_train, y_mini_train = create_balanced_mini_dataset(train_loader, samples_per_class=50)
    X_mini_test, y_mini_test = create_balanced_mini_dataset(test_loader, samples_per_class=20)
    
    print(f"Mini train set shape: {X_mini_train.shape}")
    print(f"Mini test set shape: {X_mini_test.shape}")
    
    # Save datasets
    os.makedirs('./processed_data', exist_ok=True)
    np.savez('./processed_data/mnist_subset.npz', 
             X_train=X_train_subset, y_train=y_train_subset,
             X_test=X_test_subset, y_test=y_test_subset)
    
    np.savez('./processed_data/mnist_mini.npz',
             X_train=X_mini_train, y_train=y_mini_train,
             X_test=X_mini_test, y_test=y_mini_test)
    
    print("Datasets saved to 'processed_data' directory")

###########################################
# Part 4: PyTorch Neural Network Implementation
###########################################

class PyTorchNeuralNetwork(nn.Module):
    """A simple neural network implemented with PyTorch."""
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):
        super(PyTorchNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ConvNet(nn.Module):
    """A convolutional neural network implemented with PyTorch."""
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # If input is flattened (784), reshape it to image format (1,28,28)
        if x.dim() == 2 and x.size(1) == 784:
            x = x.view(-1, 1, 28, 28)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def train_pytorch_model(model, dataloader, criterion, optimizer, device, num_epochs=5):
    """Train a PyTorch model."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch is not available. Cannot train PyTorch model.")
        return None, None, None

    model.train()
    losses = []
    accuracies = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_accuracy = correct / total * 100
        
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return losses, accuracies, training_time

def evaluate_pytorch_model(model, dataloader, device):
    """Evaluate a PyTorch model."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch is not available. Cannot evaluate PyTorch model.")
        return None, None, None
        
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy, all_preds, all_labels

def train_and_compare_with_pytorch():
    """Train and compare custom and PyTorch implementations."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch is not available. Cannot run comparison.")
        return
        
    print("\nComparing custom neural network with PyTorch implementation...")
    
    # Load data
    try:
        data = np.load('./processed_data/mnist_mini.npz')
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
    except FileNotFoundError:
        print("MNIST dataset not found. Run explore_mnist_dataset() first.")
        print("Using small random dataset for demonstration...")
        # Create small random dataset
        X_train = np.random.randn(500, 784)
        y_train = np.random.randint(0, 10, 500)
        X_test = np.random.randn(100, 784)
        y_test = np.random.randint(0, 10, 100)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Prepare PyTorch data
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 1. Train custom neural network
    print("\nTraining custom neural network from scratch...")
    custom_nn = MultiLayerNeuralNetwork([784, 128, 64, 10])
    custom_losses, custom_accuracies = custom_nn.train(
        X_train, y_train, learning_rate=0.01, epochs=20, batch_size=32, print_every=5)
    
    # Evaluate custom model
    start_time = time.time()
    custom_predictions = custom_nn.predict(X_test)
    custom_test_accuracy = np.mean(custom_predictions == y_test) * 100
    custom_time = time.time() - start_time
    print(f"Custom model test accuracy: {custom_test_accuracy:.2f}%")
    
    # 2. Train PyTorch model
    print("\nTraining PyTorch model...")
    pytorch_model = PyTorchNeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(pytorch_model.parameters(), lr=0.01, momentum=0.9)
    
    pytorch_losses, pytorch_accuracies, pytorch_time = train_pytorch_model(
        pytorch_model, train_loader, criterion, optimizer, device, num_epochs=20)
    
    # Evaluate PyTorch model
    pytorch_test_accuracy, pytorch_predictions, test_labels = evaluate_pytorch_model(
        pytorch_model, test_loader, device)
    
    # Compare results
    print("\nPerformance Comparison:")
    print(f"{'Model':<20} {'Test Accuracy (%)':<20}")
    print(f"{'-'*40}")
    print(f"{'Custom Neural Net':<20} {custom_test_accuracy:<20.2f}")
    print(f"{'PyTorch Model':<20} {pytorch_test_accuracy:<20.2f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(custom_losses, 'b-', label='Custom Implementation')
    plt.plot(pytorch_losses, 'r-', label='PyTorch Implementation')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(custom_accuracies, 'b-', label='Custom Implementation')
    plt.plot(pytorch_accuracies, 'r-', label='PyTorch Implementation')
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize predictions
    # Select random test examples to display
    num_samples = min(5, len(X_test))
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[idx]}")
        plt.axis('off')
        
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.bar(['Custom', 'PyTorch'], [custom_predictions[idx], pytorch_predictions[idx]], 
                color=['blue', 'red'])
        plt.title(f"C: {custom_predictions[idx]}, PT: {pytorch_predictions[idx]}")
    
    plt.suptitle('Sample Predictions Comparison')
    plt.tight_layout()
    plt.show()

###########################################
# Main function to run the entire workshop
###########################################

def main():
    print("=" * 70)
    print("Neural Networks Workshop: Consolidated Script")
    print("=" * 70)
    
    while True:
        print("\nPlease select an option:")
        print("1. Introduction and Activation Functions")
        print("2. Test Neural Network on XOR Problem")
        print("3. Explore MNIST Dataset")
        print("4. Train and Compare Custom vs. PyTorch Models on MNIST")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            print("\n--- Introduction and Activation Functions ---")
            plot_activation_functions()
        
        elif choice == '2':
            print("\n--- Testing Neural Network on XOR Problem ---")
            test_xor_problem()
        
        elif choice == '3':
            print("\n--- Exploring MNIST Dataset ---")
            explore_mnist_dataset()
        
        elif choice == '4':
            print("\n--- Train and Compare Models on MNIST ---")
            train_and_compare_with_pytorch()
        
        elif choice == '5':
            print("\nThank you for using the Neural Networks Workshop script!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
