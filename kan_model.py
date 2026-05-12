import torch
from kan import KAN
import time
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
