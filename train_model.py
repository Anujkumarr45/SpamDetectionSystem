import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import os

# 1. Dataset load
data = pd.read_csv("dataset/SMSSpamCollection", sep="\t", names=["label", "message"])

# 2. Encode labels (ham=0, spam=1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 3. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# 4. Convert text → numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# 7. Ensure model folder exists
os.makedirs("model", exist_ok=True)

# 8. Save model, vectorizer, and accuracy
pickle.dump(model, open("model/spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
pickle.dump(accuracy, open("model/accuracy.pkl", "wb"))

print("🎉 Model training complete and files saved!")
