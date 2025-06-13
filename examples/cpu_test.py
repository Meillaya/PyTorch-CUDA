#!/usr/bin/env python3
"""
Simple CPU-only test for MiniTorch
"""

import pyminitorch as mt

def test_basic_operations():
    print("Testing basic tensor operations (CPU only)...")
    
    # Create CPU tensors by specifying device
    try:
        a = mt.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2], mt.Device.CPU)
        b = mt.Tensor([2.0, 2.0, 2.0, 2.0], [2, 2], mt.Device.CPU)
        
        print("Tensor a:")
        a.print()
        print("Tensor b:")  
        b.print()
        
        # Test addition
        c = a.add(b)
        print("a + b:")
        c.print()
        
        # Test multiplication
        d = a.mul(b)
        print("a * b:")
        d.print()
        
        print("Basic operations test passed!")
        
    except Exception as e:
        print(f"Error in basic operations: {e}")
        return False
        
    return True

if __name__ == "__main__":
    test_basic_operations() 