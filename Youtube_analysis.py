import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

# Load the dataset
file_path = r"C:\\Users\\Vedhiga V.B\\OneDrive\\Desktop\\project2025\\sentiment analysis\\YoutubeCommentsDataSet.csv"

df = pd.read_csv(file_path, encoding="utf-8")

# Display basic information
print(df.info())
print(df.head())

# Remove missing values
df.dropna(inplace=True)

# Convert sentiment to lowercase for consistency
df["Sentiment"] = df["Sentiment"].str.lower()

# 1️⃣ Sentiment Distribution
plt.figure(figsize=(8,5))
sns.countplot(x="Sentiment", data=df, palette="coolwarm")
plt.title("Sentiment Distribution in YouTube Comments")
plt.show()

# 2️⃣ Most Common Words in Positive Comments
positive_comments = " ".join(df[df["Sentiment"] == "positive"]["Comment"])
positive_words = Counter(positive_comments.split())

# Generate word cloud for positive comments
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(positive_comments)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Positive Comments")
plt.show()

# 3️⃣ Most Common Words in Negative Comments
negative_comments = " ".join(df[df["Sentiment"] == "negative"]["Comment"])
negative_words = Counter(negative_comments.split())

# Generate word cloud for negative comments
wordcloud = WordCloud(width=800, height=400, background_color="black").generate(negative_comments)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Negative Comments")
plt.show()

# 4️⃣ Topic Modeling (LDA)
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

df['Cleaned_Comment'] = df['Comment'].apply(preprocess_text)

# Convert text data into a matrix of token counts
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Comment'])

# Apply LDA for topic modeling
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Display top words in each topic
def display_topics(model, feature_names, num_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx+1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]))
        print("\n")

display_topics(lda, vectorizer.get_feature_names_out(), 10)

# Save cleaned dataset
df.to_csv("C:\\Users\\Vedhiga V.B\\OneDrive\\Desktop\\project2025\\sentiment analysis\\Cleaned_YoutubeComments.csv", index=False)
