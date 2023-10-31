# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import the dataset
df_train = pd.read_csv("/Users/asif/Downloads/archive (8)/fashion-mnist_train.csv")
df_test = pd.read_csv("/Users/asif/Downloads/archive (8)/fashion-mnist_test.csv")
df = pd.concat([df_train, df_test], axis=0)
X = df.drop('label', axis=1)
y = df['label']

# Define classes
classes = ['tshirt', 'trouser', 'pullover', 'dress', 'coat',
           'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X / 255, y, test_size=.2, random_state=42)

# Create the model
def get_model():
    model = Sequential()
    model.add(Flatten(input_shape=(X.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

model = get_model()

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))

# Additional insights
insights = "This model achieved an accuracy of {:.2f}% on the test data.".format(accuracy * 100)

# Save results to output.txt
with open('output.txt', 'w') as f:
    f.write("Model Architecture Summary:\n")
    model.summary(print_fn=lambda x: f.write(x + '\n\n'))
    f.write("Accuracy: {:.4f}\n".format(accuracy))
    f.write("\nAdditional Insights:\n")
    f.write(insights)

# Print confirmation message
print("Model evaluation results saved to output.txt.")
