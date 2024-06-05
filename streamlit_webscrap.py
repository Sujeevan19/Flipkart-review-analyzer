import streamlit as st
import requests
from bs4 import BeautifulSoup
import unicodedata
import re
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import nltk
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.title('Flipkart Product Sentiment Analysis')


# Function to remove emojis from text
def remove_emojis(text):
    return ''.join(char for char in text if unicodedata.name(char).startswith('LATIN'))


# Function to extract only numbers and text
def extract_numbers_and_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


# Function to ensure space between every word
def ensure_space(text):
    return ' '.join(text.split())


# Function to scrape reviews from Flipkart
def scrape_reviews(url):
    all_reviews = []
    for page_num in range(1, 3):
        url_source = url
        url = url_source.format(page_num=page_num)
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        div_reviews = [div.text.strip() for div in soup.find_all('div', class_='ZmyHeo')]
        all_reviews.extend(div_reviews)
    return all_reviews


# Function to perform sentiment analysis
def analyze_sentiment(reviews):
    stop_words = set(stopwords.words('english'))
    vader = SentimentIntensityAnalyzer()
    pattern = r'[^A-Za-z0-9\s]+'

    output_df = pd.DataFrame(
        columns=['ID', 'REVIEWS', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'SENTIMENT', 'SUBJECTIVITY SCORE'])

    for idx, review in enumerate(reviews, start=1):
        cleaned_text = re.sub(pattern, ' ', review)
        words = nltk.word_tokenize(cleaned_text)
        words = [word.lower() for word in words if word.lower() not in stop_words]
        cleaned_text = ' '.join(words)
        scores = vader.polarity_scores(cleaned_text)
        scores['compound'] = scores['pos'] + scores['neg']

        blob = TextBlob(cleaned_text)
        polarity_score = blob.sentiment.polarity
        subjectivity_score = blob.sentiment.subjectivity

        output_df = pd.concat([output_df, pd.DataFrame(
            {'ID': idx, 'REVIEWS': review, 'POSITIVE SCORE': scores['pos'], 'NEGATIVE SCORE': scores['neg'],
             'SENTIMENT': scores['compound'], 'SUBJECTIVITY SCORE': subjectivity_score}, index=[0])])

    return output_df


# User input
url1 = st.text_input("Enter your Flipkart Product URL")
if st.button("Analyze Sentiment"):
    if url1:
        # Scrape reviews
        all_reviews = scrape_reviews(url1)

        # Analyze sentiment
        output_df = analyze_sentiment(all_reviews)

        # Visualizations
        # Histogram plot
        diagram1 = sns.histplot(output_df['SENTIMENT'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        st.pyplot(diagram1.figure)

        # Bar plot for overall scores
        plt.figure(figsize=(6, 4))
        plt.bar(['POSITIVE SCORE', 'NEGATIVE SCORE'],
                [output_df['POSITIVE SCORE'].sum(), output_df['NEGATIVE SCORE'].sum()])
        plt.title('Overall Scores')
        plt.xlabel('Score Type')
        plt.ylabel('Score')
        st.pyplot(plt.gcf())

        # Word cloud plot
        positive_reviews = output_df[output_df['POSITIVE SCORE'] > output_df['NEGATIVE SCORE']]
        positive_test = ' '.join(review for review in positive_reviews['REVIEWS'])
        positive_wordcloud = WordCloud(width=800, height=600, background_color='white').generate(positive_test)
        plt.figure(figsize=(10, 5))
        plt.imshow(positive_wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt.gcf())
