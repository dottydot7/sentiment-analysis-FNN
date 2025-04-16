import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('movie_data.csv')
print(df.head())
print(df.shape)

nltk.download('stopwords')
stop = stopwords.words('english')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

porter = PorterStemmer()

df['review'] = df['review'].apply(preprocessor)

X = df['review'].values
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"Train shape: {X_train_tfidf.shape}")
print(f"Test shape: {X_test_tfidf.shape}")
