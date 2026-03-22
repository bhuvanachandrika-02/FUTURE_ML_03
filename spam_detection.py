import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    'Message': [
        'Win money now',
        'Hello how are you',
        'Claim your prize',
        'Let’s meet tomorrow',
        'Free gift available',
        'Call me later'
    ],
    'Label': [1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Message'])

y = df['Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test prediction
test_msg = ["Win a free iPhone now"]
test_data = vectorizer.transform(test_msg)

prediction = model.predict(test_data)

print("Message:", test_msg[0])
print("Prediction:", "Spam" if prediction[0]==1 else "Not Spam")