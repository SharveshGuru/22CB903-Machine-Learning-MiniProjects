import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import re
import string
from gensim.models import Word2Vec
import numpy as np

# Data preprocessing function: clean and tokenize the text
def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove digits
    return text

# Load the dataset
dataset = pd.read_csv(r"C:\stuff\college stuff\study\cb903\22CB903-Machine-Learning-MiniProjects\02-Text Classification\dataset.csv")

# Preprocess the text messages
dataset['cleaned_message'] = dataset['Message'].apply(preprocess_text)

# Encode the labels: ham -> 0, spam -> 1
label_encoder = LabelEncoder()
dataset['label'] = label_encoder.fit_transform(dataset['Category'])

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    dataset['cleaned_message'], dataset['label'], test_size=0.2, random_state=42
)

# Step 2: TF-IDF Feature Engineering
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 3: Train a Logistic Regression model on TF-IDF features
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Make predictions and evaluate the model
y_pred_tfidf = lr_model.predict(X_test_tfidf)
report_tfidf = classification_report(y_test, y_pred_tfidf, target_names=['ham', 'spam'])
print("TF-IDF Model Performance:\n", report_tfidf)

# Step 4: Word Embedding (Word2Vec) Feature Engineering
# Tokenize the cleaned text for Word2Vec input
tokenized_messages = dataset['cleaned_message'].apply(lambda x: x.split())

# Train a Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_messages, vector_size=100, window=5, min_count=1, workers=4)

# Function to get the average word vector for a sentence
def get_avg_word2vec(sentence, model, vector_size):
    words = sentence.split()
    avg_vector = np.zeros(vector_size)
    valid_words = 0
    for word in words:
        if word in model.wv:
            avg_vector += model.wv[word]
            valid_words += 1
    if valid_words > 0:
        avg_vector /= valid_words
    return avg_vector

# Transform the dataset using the average word vectors
X_train_w2v = np.array([get_avg_word2vec(sentence, word2vec_model, 100) for sentence in X_train])
X_test_w2v = np.array([get_avg_word2vec(sentence, word2vec_model, 100) for sentence in X_test])

# Train a Logistic Regression model on Word2Vec features
lr_w2v_model = LogisticRegression(max_iter=1000)
lr_w2v_model.fit(X_train_w2v, y_train)

# Make predictions and evaluate the Word2Vec model
y_pred_w2v = lr_w2v_model.predict(X_test_w2v)
report_w2v = classification_report(y_test, y_pred_w2v, target_names=['ham', 'spam'])
print("\nWord2Vec Model Performance:\n", report_w2v)
