import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
import random


csv_file = r"C:\Users\saiva\OneDrive\Desktop\pragament\handwritten_alphabets.csv"
data = pd.read_csv(csv_file)


y = data.iloc[:, 0].values


X = data.iloc[:, 1:].values


X = X / 255.0

X = X.reshape(-1, 28, 28, 1)


_, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



model_path = r"C:\Users\saiva\OneDrive\Desktop\pragament\mnist-99-tensorflow2-default-v1\best_model.h5"  # Adjust if needed
model = load_model(model_path)
print("✅ Loaded MNIST model successfully.")



predictions = model.predict(X_test)

# Display 10 predictions
print("\n🔍 Displaying predictions on A-Z characters using MNIST digit model:\n")

for i in range(10):
    idx = random.randint(0, len(X_test)-1)
    img = X_test[idx].reshape(28, 28)
    actual_char = y_test[idx]

    predicted_digit = np.argmax(predictions[idx])

    plt.imshow(img, cmap='gray')
    plt.title(f"Actual: {actual_char} | Predicted Digit: {predicted_digit}")
    plt.axis("off")
    plt.show()
