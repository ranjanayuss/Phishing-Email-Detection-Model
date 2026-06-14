import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("emails.csv")

# Features and labels
X = data["email"]
y = data["label"]

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(5,4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Safe","Phishing"])
plt.yticks([0,1], ["Safe","Phishing"])

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j],
                 ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# User Prediction
print("\n----- Email Scanner -----")

email_text = input("Enter email text: ")

email_vector = vectorizer.transform([email_text])

result = model.predict(email_vector)

if result[0] == "phishing":
    print("⚠️ Phishing Email Detected")
else:
    print("✅ Safe Email")
