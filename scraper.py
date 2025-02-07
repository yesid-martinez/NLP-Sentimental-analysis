import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load data from CSV file
file_path = 'comments.csv' # Make sure the file is in the same directory or provide the full path
df = pd.read_csv(file_path)

# Check if “Comment” column exists
df = df.dropna(subset=['Comment']) # Remove rows with empty comments

# Apply sentiment analysis to each comment
df['sentiment_score'] = df['Comment'].apply(lambda comment: sia.polarity_scores(comment)['compound'])

# Calculate the average sentiment of the first 200 comments
num_comments = min(200, len(df)) # Take 200 or the total if less.
average_sentiment = df['sentiment_score'][:num_comments].mean()

# Mostrar los resultados
print(df[['Comment', 'sentiment_score']].head(200))
print(f'Promedio de sentimiento de los primeros {num_comments} comentarios: {average_sentiment:.4f}')

# Interpretation of the sentiment_score:
# - Greater than 0 → Positive sentiment (e.g. 0.8442 is a clearly positive comment).
# - Less than 0 → Negative sentiment (e.g. -0.2500 indicates that the comment is negatively charged).
# Close to 0 → Neutral sentiment (e.g. 0.0000).