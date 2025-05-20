# Digit-Recognition-system
# 🧠 Digit Recognition System (Scikit-learn)

This project is a **Handwritten Digit Recognition System** built using traditional Machine Learning techniques (Logistic Regression) and the **scikit-learn** library. It classifies digits (0–9) from grayscale images using the built-in `load_digits()` dataset from `sklearn.datasets`.

---

## 📌 Features

- Recognizes handwritten digits from images
- Built using **Logistic Regression** (no deep learning or TensorFlow)
- Accuracy over **95%** on test data
- Visualizes digit predictions
- Clean and minimal setup

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Scikit-learn**
- **Matplotlib**
- **NumPy**

---

## 📂 Dataset

- Uses `load_digits()` from `sklearn.datasets`
- Contains **1797** labeled samples of 8x8 grayscale images of digits

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/digit-recognition-sklearn.git
cd digit-recognition-sklearn
```
2. Install Dependencies

pip install -r requirements.txt
Or manually install:

pip install numpy scikit-learn matplotlib
3. Run the Code

python digit_recognition.py
You will see sample digits, accuracy scores, and visual predictions.

## 📈 Sample Output

✅ Accuracy: 0.9666

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        38
           1       0.95      1.00      0.97        33
           2       1.00      0.94      0.97        36
           ...
## 🔍 Visual Preview

Predicted: 3, Actual: 3


Predicted: 8, Actual: 8

## 📦 File Structure

digit-recognition-sklearn/
│
├── digit_recognition.py       # Main project script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── preview1.png, preview2.png # Optional image samples
