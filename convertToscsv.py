import os
import numpy as np
import pandas as pd
from PIL import Image


train_path = r'C:\Users\saiva\OneDrive\Desktop\pragament\archive\alphabet-dataset\Train'
val_path = r'C:\Users\saiva\OneDrive\Desktop\pragament\archive\alphabet-dataset\Validation'

img_size = (28, 28)
data = []

def process_dataset(dataset_dir):
    for label in sorted(os.listdir(dataset_dir)):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(img_size)
                img_array = np.array(img).flatten()
                row = [label] + img_array.tolist()
                data.append(row)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


process_dataset(train_path)
process_dataset(val_path)


columns = ['label'] + [f'pixel{i}' for i in range(img_size[0] * img_size[1])]
df = pd.DataFrame(data, columns=columns)
df.to_csv('handwritten_alphabets.csv', index=False)
print("CSV file saved successfully!")
