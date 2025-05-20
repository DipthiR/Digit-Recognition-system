# Import libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load the dataset
digits = load_digits()

# Step 2: Explore the data
print("Image data shape:", digits.images.shape)
print("Label data shape:", digits.target.shape)

# Step 3: Visualize some digits
for i in range(5):
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")
    plt.axis('off')
    plt.show()

# Step 4: Flatten the image data (8x8 → 64)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Step 5: Split into train and test
x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, random_state=42)

# Step 6: Train a classifier (Logistic Regression)
model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(x_test)

# Step 8: Evaluate the model
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Show some predictions
for i in range(5):
    plt.imshow(x_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.show()
