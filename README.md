# 🎯 Sentiment Analysis on YouTube Comments

This project performs **sentiment analysis** on YouTube comments using **Natural Language Processing (NLP)** and **machine learning**. It classifies comments into **Positive**, **Negative**, or **Neutral** categories and provides visual insights using word clouds and charts.

---

## 📁 Dataset

The dataset consists of YouTube video comments along with manually labeled sentiments.  
Format:
- `Comment`: Text of the comment  
- `Sentiment`: One of `positive`, `negative`, or `neutral`

> 🔄 You can use web scraping or YouTube APIs to extract your own dataset.

---

## 🔍 Project Workflow

1. **Data Cleaning & Preprocessing**
   - Remove stopwords, punctuation
   - Tokenization using `nltk`
   - Lowercasing and lemmatization
2. **Exploratory Data Analysis (EDA)**
   - Sentiment distribution bar chart
   - Word clouds for positive & negative comments
   - Topic modeling using LDA
3. **Feature Extraction**
   - TF-IDF Vectorizer
4. **Model Building**
   - Logistic Regression (can be replaced with XGBoost/SVM)
   - Evaluation: Accuracy, Precision, Recall, F1-score
5. **Deployment (Optional)**
   - Flask app to predict sentiment of user-input comments

---

## 🧠 Libraries Used

```
pandas
numpy
nltk
matplotlib
seaborn
sklearn
wordcloud
flask (optional for deployment)
```
