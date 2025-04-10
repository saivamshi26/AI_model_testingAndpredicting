import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Step 1: Load CSV Dataset ===
csv_path = r"C:\Users\saiva\OneDrive\Desktop\pragament\handwritten_alphabets.csv"
df = pd.read_csv(csv_path)

# === Step 2: Split features and labels ===
X = df.drop('label', axis=1).values
y = df['label'].values  # labels are already A–Z

# === Step 3: Normalize and reshape images ===
X = X / 255.0
X = X.reshape(-1, 28, 28)

# === Step 4: Load Pretrained MNIST Model ===
model_path = r"C:\Users\saiva\OneDrive\Desktop\pragament\mnist-99-tensorflow2-default-v1\best_model.h5"
model = load_model(model_path)
<<<<<<< HEAD
print("✅ MNIST model loaded successfully.")
=======
print(" Loaded MNIST model successfully.")
>>>>>>> d8a297952289624ee496640132b468060ebbcece

# === Step 5: Ask how many predictions to make ===
try:
    num_preds = int(input("🔢 How many predictions would you like to see? "))
    num_preds = min(num_preds, len(X))  # limit to dataset size
except ValueError:
    print("❌ Please enter a valid number.")
    exit()

# === Step 6: Predict and show results ===
for idx in range(num_preds):
    image = X[idx]
    prediction = model.predict(image.reshape(1, 28, 28, 1))
    predicted_digit = np.argmax(prediction)

<<<<<<< HEAD
    plt.imshow(image, cmap='gray')
    plt.title(f"Actual: {y[idx]} | Predicted Digit: {predicted_digit}")
    plt.axis('off')
=======
predictions = model.predict(X_test)

# Display 10 predictions
print("\n Displaying predictions on A-Z characters using MNIST digit model:\n")

for i in range(10):
    idx = random.randint(0, len(X_test)-1)
    img = X_test[idx].reshape(28, 28)
    actual_char = y_test[idx]

    predicted_digit = np.argmax(predictions[idx])

    plt.imshow(img, cmap='gray')
    plt.title(f"Actual: {actual_char} | Predicted Digit: {predicted_digit}")
    plt.axis("off")
>>>>>>> d8a297952289624ee496640132b468060ebbcece
    plt.show()
