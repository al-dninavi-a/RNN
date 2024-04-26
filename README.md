# RNN
#Parkinson disease 
# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

# Load the dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data')

# Split the dataset into features and labels
X = df.drop(['name', 'status'], axis=1).values
y = df['status'].values

# Scale the features to the range [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to fixed-point numbers
X_train = np.fix(X_train * (2**5))
X_test = np.fix(X_test * (2**5))

# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu', input_shape=(128,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Start measuring the execution time
start_time = time.time()

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./drive/MyDrive/model_chkp',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=10, callbacks=[model_checkpoint_callback])
# Train the model
#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

# Stop measuring the execution time
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

# Evaluate the model
t1 = time.time()
test_loss, test_acc = model.evaluate(X_test, y_test)

# Calculate recognition time
t2 = time.time()
prediction_time = t2 - t1
print("Prediction time: %s seconds ---" % (prediction_time))

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
print(f'Execution Time: {execution_time} seconds')

import matplotlib.pyplot as plt
# Plot the accuracy and loss curves
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
