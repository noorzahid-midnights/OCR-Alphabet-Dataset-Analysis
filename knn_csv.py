import os
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import joblib

# ---------------------- Paths -------------------------------
csv_folder_path = "training_csv"
testing_file_path = "E:\\Python\\OCR Project\\template"
model_path = "knn_combined_model_csv.pkl"

# ---------------------- Combine All CSVs -------------------------------
print(" Loading and combining all CSV files...\n")
all_dataframes = []

for csv_file in os.listdir(csv_folder_path):
    if csv_file.lower().endswith(".csv"):
        csv_path = os.path.join(csv_folder_path, csv_file)
        df = pd.read_csv(csv_path, dtype={'Label': str})
        all_dataframes.append(df)
        print(f" Loaded '{csv_file}'")

if not all_dataframes:
    print(" No CSV files found. Exiting.")
    exit()

combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f"\n Total samples combined: {len(combined_df)}")

training_labels = combined_df['Label'].values
training_features = combined_df.drop(columns=['Label']).values

# ---------------------- Train/Test Split -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    training_features, training_labels, test_size=0.2, random_state=42
)

# ---------------------- Train Model -------------------------------
print("\n Training combined model...\n")
knn = KNeighborsClassifier(n_neighbors=3)
model = knn.fit(X_train, y_train)
print(" Model trained.")

# ---------------------- Save Model -------------------------------
joblib.dump(model, model_path)
print(f" Model saved as '{model_path}'")

# ---------------------- Predict from Test Images -------------------------------
print("\n Predicting from test images...\n")
predicted_labels = []
true_labels = []

for root, _, files in os.walk(testing_file_path):
    for file in files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        file_full_path = os.path.join(root, file)
        img = cv2.imread(file_full_path)

        if img is None:
            print(f" Failed to load image: {file_full_path}")
            continue

        # Image Preprocessing
        img_resized = cv2.resize(img, (28, 28))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 2)
        _, img_thresh = cv2.threshold(img_blur, 128, 255, cv2.THRESH_BINARY)

        flattened_test_img = img_thresh.flatten() / 255.0
        prediction = model.predict([flattened_test_img])[0]
        true_label = os.path.splitext(file)[0].upper()

        predicted_labels.append(prediction)
        true_labels.append(true_label)

        print(f"{file} => Predicted: {prediction}, Actual: {true_label}")

# ---------------------- Accuracy Report -------------------------------
if predicted_labels and true_labels:
    print("\n📄 Classification Report:")
    print(classification_report(true_labels, predicted_labels))

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    print("\n Metrics:")
    print(f" Accuracy : {accuracy * 100:.2f}%")
    print(f" Precision: {precision * 100:.2f}%")
    print(f" Recall   : {recall * 100:.2f}%")
    print(f" F1 Score : {f1 * 100:.2f}%")
else:
    print("\n No valid test images found to evaluate accuracy.")

cv2.destroyAllWindows()
