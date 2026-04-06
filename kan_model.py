'''
import torch
from kan import KAN
import time

def train_kan(X_train, y_train, input_dim, output_dim):
    # KAN architecture: [input, hidden_layer, output]
    # KANs need fewer neurons than MLPs for similar tasks
    model = KAN(width=[input_dim, 5, output_dim], grid=5, k=3) 
    
    start_time = time.time()
    # KANs typically use LBFGS for scientific tasks, but Adam is better for high-dim
    results = model.train({'train_input': X_train, 'train_label': y_train}, mode=True) 
                          #,opt="Adam", steps=20, lr=0.01)
    
    end_time = time.time()
    print(f"KAN Training Time: {end_time - start_time:.2f} seconds")
    return model
'''    
import torch
from kan import KAN
import time
'''
def train_kan(X_train, y_train, input_dim, output_dim):
    # 1. Ensure 2D shape: (samples, features)
    if X_train.ndim == 1:
        X_train = X_train.unsqueeze(1)
    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(1)

    # 2. Initialize Model
    # width=[input, hidden, output], grid=number of intervals, k=spline order
    model = KAN(width=[input_dim, 5, output_dim], grid=5, k=3) 
    
    # 3. Construct the mandatory dataset dictionary
    dataset = {
        'train_input': X_train,
        'train_label': y_train,
        'test_input': X_train,  # Original pykan requires these keys
        'test_label': y_train
    }
    
    start_time = time.time()
    
    # 4. Training
    # Note: 'steps' in pykan refers to the number of LBFGS/Adam iterations
    results = model.train(dataset, opt="Adam", steps=20, lr=0.01)
    
    end_time = time.time()
    print(f"KAN Training Time: {end_time - start_time:.2f} seconds")
    return model
    
def train_kan(X_train, y_train, input_dim, output_dim):
    model = KAN(width=[input_dim, 5, output_dim], grid=5, k=3) 
    
    dataset = {
        'train_input': X_train,
        'train_label': y_train,
        'test_input': X_train,
        'test_label': y_train
    }
    
    # Try 'opt_name' if 'optimizer' fails, or vice versa
    results = model.train(dataset, opt_name="Adam", steps=20, lr=0.01)
    
    return model'''
    
def train_kan(X_train, y_train, input_dim, output_dim):
    # Ensure dataset is in the correct dictionary format
    dataset = {
        'train_input': X_train,
        'train_label': y_train,
        'test_input': X_train,
        'test_label': y_train
    }

    model = KAN(width=[input_dim, 5, output_dim], grid=5, k=3) 
    
    start_time = time.time()
    
    # In original pykan, the method is .fit(), not .train()
    # The argument for the optimizer is usually 'opt'
    results = model.fit(dataset, opt="Adam", steps=20, lr=0.01)
    
    end_time = time.time()
    print(f"KAN Training Time: {end_time - start_time:.2f} seconds")
    return model