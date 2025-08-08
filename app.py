from flask import Flask, request, render_template
import pickle
import re

# Load model and vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = ""
    if request.method == "POST":
        comment = request.form["comment"]
        cleaned = clean_text(comment)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        sentiment = f"Predicted Sentiment: {prediction.capitalize()}"
    return render_template("index.html", result=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
