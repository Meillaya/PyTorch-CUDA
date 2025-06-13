#!/usr/bin/env python3
"""
Simple example demonstrating MiniTorch usage
"""

import pyminitorch as mt

def main():
    print("MiniTorch Simple Example")
    print("=" * 40)
    
    # Create tensors
    print("Creating tensors...")
    a = mt.ones([3, 4])
    b = mt.zeros([3, 4])
    c = mt.randn([3, 4])
    
    print("Tensor a:")
    a.print()
    print("Tensor b:")
    b.print()
    print("Tensor c:")
    c.print()
    
    # Basic operations
    print("\nPerforming basic operations...")
    d = a.add(c)
    print("a + c:")
    d.print()
    
    e = a.mul(c)
    print("a * c:")
    e.print()
    
    # Matrix operations
    print("\nMatrix operations...")
    x = mt.randn([4, 3])
    y = a.matmul(x)
    print("Matrix multiplication result:")
    y.print()
    
    # Activation functions
    print("\nActivation functions...")
    relu_result = c.relu()
    print("ReLU of c:")
    relu_result.print()
    
    sigmoid_result = c.sigmoid()
    print("Sigmoid of c:")
    sigmoid_result.print()
    
    # Gradient computation
    print("\nGradient computation...")
    a.requires_grad = True
    c.requires_grad = True
    
    loss = (a.mul(c)).sum()
    print("Loss:", loss)
    loss.backward()
    
    print("Gradient of a:")
    if a.grad:
        a.grad.print()
    
    print("Gradient of c:")
    if c.grad:
        c.grad.print()

if __name__ == "__main__":
    main() 