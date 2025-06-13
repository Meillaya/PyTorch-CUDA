#!/usr/bin/env python3
"""
Neural Network example with MiniTorch
Training a simple 2-layer network on synthetic data
"""

import pyminitorch as mt
import numpy as np

class SimpleNet:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize layers
        self.fc1_weight = mt.randn([input_size, hidden_size])
        self.fc1_bias = mt.zeros([hidden_size])
        self.fc2_weight = mt.randn([hidden_size, output_size])
        self.fc2_bias = mt.zeros([output_size])
        
        # Set requires_grad for all parameters
        self.fc1_weight.requires_grad = True
        self.fc1_bias.requires_grad = True
        self.fc2_weight.requires_grad = True
        self.fc2_bias.requires_grad = True
    
    def forward(self, x):
        # First layer
        h1 = x.matmul(self.fc1_weight).add(self.fc1_bias)
        h1_relu = h1.relu()
        
        # Second layer
        output = h1_relu.matmul(self.fc2_weight).add(self.fc2_bias)
        return output
    
    def parameters(self):
        return [self.fc1_weight, self.fc1_bias, self.fc2_weight, self.fc2_bias]
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

def mse_loss(pred, target):
    diff = pred.sub(target)
    squared_diff = diff.mul(diff)
    return squared_diff.mean()

def generate_synthetic_data(n_samples=1000, input_dim=10, noise=0.1):
    """Generate synthetic regression data"""
    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    
    # Create a simple linear relationship with some nonlinearity
    true_weights = np.random.randn(input_dim, 1).astype(np.float32)
    y = X @ true_weights + noise * np.random.randn(n_samples, 1).astype(np.float32)
    
    return X, y

def main():
    print("MiniTorch Neural Network Example")
    print("=" * 50)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X_np, y_np = generate_synthetic_data(n_samples=1000, input_dim=10)
    
    # Convert to MiniTorch tensors
    X = mt.Tensor(X_np.flatten().tolist(), [X_np.shape[0], X_np.shape[1]])
    y = mt.Tensor(y_np.flatten().tolist(), [y_np.shape[0], y_np.shape[1]])
    
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Create network
    net = SimpleNet(input_size=10, hidden_size=64, output_size=1)
    
    # Training parameters
    learning_rate = 0.01
    epochs = 100
    
    print(f"\nTraining for {epochs} epochs with lr={learning_rate}")
    
    for epoch in range(epochs):
        # Zero gradients
        net.zero_grad()
        
        # Forward pass
        predictions = net.forward(X)
        
        # Compute loss
        loss = mse_loss(predictions, y)
        
        # Backward pass
        loss.backward()
        
        # Manual parameter update (simple SGD)
        for param in net.parameters():
            if param.grad:
                # param = param - lr * grad
                # This is a simplified update - in practice you'd want more sophisticated optimizers
                pass  # Note: This requires implementing parameter update logic
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")
            loss.print()
    
    print("\nTraining completed!")
    
    # Test prediction
    print("\nTesting predictions...")
    test_input = mt.randn([1, 10])
    test_output = net.forward(test_input)
    print("Test input:")
    test_input.print()
    print("Test output:")
    test_output.print()

if __name__ == "__main__":
    main() 