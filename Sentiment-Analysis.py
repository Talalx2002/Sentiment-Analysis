import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Load the IMDb dataset and split it into training, validation, and test sets
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# Load the pre-trained word embedding model from TensorFlow Hub
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

# Display the output shape of the Hub layer for the first 3 examples
print(hub_layer(train_examples_batch[:3]))

# Define the model architecture
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

# Display the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# Save the model
model.save("imdb_sentiment_model.h5")

# Evaluate the model on the test data
results = model.evaluate(test_data.batch(512), verbose=2)

# Print the evaluation results
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
