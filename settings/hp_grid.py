# import keras_tuner as kt

# HP_HIDDEN_LAYER_SIZE = [5, 10, 20, 40, 80, 160]
# HP_DROPOUT_RATE = [0.1, 0.2, 0.3, 0.4, 0.5]
# HP_MINIBATCH_SIZE= [64, 128, 256]
# HP_LEARNING_RATE = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]  # smaller, safer learning rates

# # HP_MINIBATCH_SIZE= [32, 64, 128]
# # HP_LEARNING_RATE = [1e-4, 1e-3, 1e-2, 1e-1]
# HP_MAX_GRADIENT_NORM = [0.01, 1.0, 100.0]

# # # RANDOM_SEARCH_ALGORITHM = kt.RandomSearch



# # import keras_tuner as kt

# # # Hidden layer sizes: start smaller to avoid instability
# # HP_HIDDEN_LAYER_SIZE = [20, 40, 80]  

# # # Dropout rates: moderate to avoid exploding/vanishing gradients
# # HP_DROPOUT_RATE = [0.1, 0.2, 0.3]  

# # # Batch sizes: reasonable for memory and stability
# # HP_MINIBATCH_SIZE = [64, 128]  

# # # Learning rates: smaller, safer values
# # HP_LEARNING_RATE = [1e-4, 3e-4, 1e-3, 3e-3]
# # # HP_LEARNING_RATE = [0.0001]  

# # # Gradient clipping: avoid too small or too extreme values
# # HP_MAX_GRADIENT_NORM = [1.0, 10.0, 100.0]  

# # Optional: you can use random search
# # RANDOM_SEARCH_ALGORITHM = kt.RandomSearch

# import keras_tuner as kt

# # Chosen "Best Guess" parameters based on Paper Exhibit 14 ranges 

# # Paper range: [5, 10, 20, 40, 80, 160]
# # We choose 80 (middle-upper end for complex tasks)
# # HP_HIDDEN_LAYER_SIZE = [80] 

# # # Paper range: [0.1, 0.2, 0.3]
# # # We choose 0.1 (Standard low dropout)
# # HP_DROPOUT_RATE = [0.1]

# # # Paper range: [32, 64, 128]
# # # We choose 64 (Balanced batch size)
# # HP_MINIBATCH_SIZE = [64]

# # # Paper range: [10^-4, 10^-3, 10^-2, 10^-1]
# # # We choose 0.001 (Standard Adam default)
# # HP_LEARNING_RATE = [1e-3]

# # # Paper range: [10^-2, 10^0, 10^2]
# # # We choose 1.0 (Standard gradient clipping)
# # HP_MAX_GRADIENT_NORM = [1.0]
import keras_tuner as kt

# Reduced search space to ensure 'random' picks are still good choices
HP_HIDDEN_LAYER_SIZE = [40, 80, 160] 
HP_DROPOUT_RATE = [0.1, 0.2, 0.3] 
HP_MINIBATCH_SIZE = [64] # Fixed to 64 to protect your 3050 Ti VRAM
HP_LEARNING_RATE = [1e-3, 1e-4]
HP_MAX_GRADIENT_NORM = [1.0]