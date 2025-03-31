# import requests
# from bs4 import BeautifulSoup

# def scrape_financial_news(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     headlines = []
#     for item in soup.find_all('h3'):  # Modify tag based on website structure
#         headlines.append(item.text.strip())
    
#     return headlines

# news_url = "https://finance.yahoo.com/"
# financial_news = scrape_financial_news(news_url)
# print(financial_news[:5])  # Display first 5 headlines

# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import string

# nltk.download('stopwords')
# nltk.download('punkt')

# def preprocess_text(text):
#     tokens = word_tokenize(text.lower())  # Convert to lowercase
#     tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
#     tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
#     return ' '.join(tokens)

# cleaned_news = [preprocess_text(news) for news in financial_news]
# print(cleaned_news[:5])  # Display cleaned headlines


# from nltk.sentiment import SentimentIntensityAnalyzer

# nltk.download('vader_lexicon')
# sia = SentimentIntensityAnalyzer()

# def analyze_sentiment(text):
#     sentiment_score = sia.polarity_scores(text)['compound']
#     if sentiment_score > 0.05:
#         return "Positive"
#     elif sentiment_score < -0.05:
#         return "Negative"
#     else:
#         return "Neutral"

# news_sentiments = [analyze_sentiment(news) for news in cleaned_news]

# # Display results
# for news, sentiment in zip(financial_news[:5], news_sentiments[:5]):
#     print(f"News: {news} \nSentiment: {sentiment}\n")


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score

# # Sample dataset
# data = [
#     ("Stock prices soar after strong earnings report", "Positive"),
#     ("Markets crash as inflation fears rise", "Negative"),
#     ("Mixed results for investors today", "Neutral"),
#     ("Tech stocks surge after positive outlook", "Positive"),
#     ("Recession fears weigh on investor sentiment", "Negative"),
# ]

# texts, labels = zip(*data)

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(texts)
# y = [1 if label == "Positive" else 0 if label == "Neutral" else -1 for label in labels]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = MultinomialNB()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")


# import streamlit as st

# st.title("Financial News Sentiment Analysis")

# # User Input
# user_input = st.text_area("Enter a financial news headline:")

# if st.button("Analyze Sentiment"):
#     cleaned_input = preprocess_text(user_input)
#     sentiment = analyze_sentiment(cleaned_input)
#     st.write(f"**Sentiment:** {sentiment}")

# # Show scraped news with sentiment
# st.subheader("Recent Financial News Sentiments")
# for news, sentiment in zip(financial_news[:5], news_sentiments[:5]):
#     st.write(f"- {news} **({sentiment})**")


import streamlit as st
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to get financial news from Google News RSS
def get_financial_news():
    url = "https://news.google.com/rss/search?q=finance&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    news_list = []
    for entry in feed.entries[:5]:  # Fetch top 5 headlines
        news_list.append(entry.title)
    
    return news_list

# Function to analyze sentiment
def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Streamlit UI
st.title("📈 Financial News Sentiment Analysis")

# User input for custom headline sentiment analysis
user_input = st.text_input("Enter a financial news headline:")
if user_input:
    sentiment = analyze_sentiment(user_input)
    st.write(f"**Sentiment:** {sentiment}")

# Display recent financial news and their sentiments
st.subheader("📰 Recent Financial News Sentiments")

financial_news = get_financial_news()

if financial_news:
    for news in financial_news:
        sentiment = analyze_sentiment(news)
        st.write(f"- {news} **({sentiment})**")
else:
    st.write("No financial news found. Try again later.")

