import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')

# Load input dataframe
df = pd.read_csv("reviews.csv", header=None, names=["REVIEWS"])

# Load stopwords
stop_words = set(stopwords.words('english'))

# Initialize sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Define regex pattern to match unwanted characters
pattern = r'[^A-Za-z\s]+'

# Create empty list to collect data for output dataframe
output_data = []

# Loop through rows of input dataframe
for index, row in df.iterrows():
    ID = index + 1  # Generate an ID based on the index
    REVIEWS = row["REVIEWS"]

    # Apply regex to remove unwanted characters
    cleaned_text = re.sub(pattern, ' ', REVIEWS)

    # Tokenize text into words
    words = nltk.word_tokenize(cleaned_text)
    
    # Remove stopwords and lowercase
    words = [word.lower() for word in words if word.lower() not in stop_words]

    # Join words back into cleaned text
    cleaned_text = ' '.join(words)

    # Get polarity scores for cleaned text
    scores = vader.polarity_scores(cleaned_text)

    # Get the subjectivity score
    blob = TextBlob(cleaned_text)
    subjectivity_score = blob.sentiment.subjectivity

    # Classify sentiment based on compound score
    if scores['compound'] > 0.05:
        sentiment_class = 'Positive'
    elif scores['compound'] < -0.05:
        sentiment_class = 'Negative'
    else:
        sentiment_class = 'Neutral'

    # Collect the data in a dictionary
    output_data.append({
        "ID": ID,
        "REVIEWS": REVIEWS,
        "POSITIVE_SCORE": scores["pos"],
        "NEGATIVE_SCORE": scores["neg"],
        "SENTIMENT": scores["compound"],
        "SUBJECTIVITY_SCORE": subjectivity_score,
        "SENTIMENT_CLASS": sentiment_class
    })

# Convert the list of dictionaries into a DataFrame
output_df = pd.DataFrame(output_data)

# Display the first few rows of the output dataframe
print(output_df.head())

# Summarize the sentiment distribution
sentiment_counts = output_df['SENTIMENT_CLASS'].value_counts()
print("\nSentiment Distribution:\n", sentiment_counts)

# Plot the sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='SENTIMENT_CLASS', data=output_df)
plt.title('Distribution of Sentiments in Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
