import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load your labeled, cleaned data
df = pd.read_csv("C:\\Users\\Vedhiga V.B\\OneDrive\\Desktop\\project2025\\sentiment analysis\\Cleaned_YoutubeComments.csv")

# Step 2: Clean again if needed
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df["Cleaned_Comment"] = df["Comment"].apply(clean_text)

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["Cleaned_Comment"])
y = df["Sentiment"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Predict and Evaluate
y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Save predictions for dashboard or app
df_test = df.iloc[y_test.index]
df_test["Predicted"] = y_pred
df_test.to_csv("Predicted_Sentiment_Output.csv", index=False)

import pickle

# Save TF-IDF Vectorizer
with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save Trained Model
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)