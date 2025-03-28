# Here is a little program that sets up a model in keras/tensorflow.
# and saves it in the .npz that can be read by simd_neuralnet.

# Then it creates a random sample vector and a random target vector
# and saves this to sample_target.npz.

# Then it uses tensorflow to do one backward calculation to calculate
# the gradient of the loss function w.r.t each parameter for the sample
# and target vector.

# This three files can then be used to check if simd_neuralnet calculates
# the same gradient as keras/tensorflow. The code that does this comparison
# is in test_backpropagation_files.c.

input_size = 292
output_size = 5
LOSS = 'binary_crossentropy'
MODEL_FILE = "model_weights.npz"
SAMPLE_FILE = "sample_target.npz"
GRADIENT_FILE = "gradient.npz"

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Define the model - Please feel free to change this to whatever you want to test.
model = keras.Sequential([
    layers.Input((input_size,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(output_size, activation='sigmoid')
])

# Compile the model - Please choose the loss function of your taste. The
# optimiser parameter is never used. 
model.compile(loss=LOSS, optimizer='adam')

# Extract weights and biases from the model - This code even shows how to
# convert a Dense network from keras to simd_neuralnet.
weights_and_biases = []
activations = []
for layer in model.layers:
	if isinstance(layer, layers.Dense):
		weights_and_biases.append(layer.get_weights()[0])  # Weights
		weights_and_biases.append(layer.get_weights()[1])  # Biases
		activations.append(layer.activation.__name__)  # Activation function

activations = np.array(activations).astype('S')
weights_and_biases.append(activations)
# Save the weights and biases as an npz file
np.savez(MODEL_FILE, *weights_and_biases)

# Generate a random sample and a random target vector (using float values)
sample = np.random.rand(input_size).astype(np.float32)
target = np.random.rand(output_size).astype(np.float32)
np.savez(SAMPLE_FILE, sample, target)

# Import tensorflow for gradient calculation
import tensorflow as tf

# Convert the sample and target to tensors
sample_tensor = tf.convert_to_tensor(sample.reshape(1, -1), dtype=tf.float32)
target_tensor = tf.convert_to_tensor(target.reshape(1, -1), dtype=tf.float32)

# Use GradientTape to calculate the gradient
with tf.GradientTape() as tape:
	output = model(sample_tensor)
	loss = tf.keras.losses.binary_crossentropy(target_tensor, output)

# Calculate the gradient
gradient = tape.gradient(loss, model.trainable_variables)
gradient_list = [np.array(g) for g in gradient]
np.savez(GRADIENT_FILE, *gradient_list )

print("A model, a random sample/target and a gradient has been calculated.\n")
print(f"./test_backpropagation_files {MODEL_FILE} {SAMPLE_FILE} {GRADIENT_FILE} {LOSS}")

