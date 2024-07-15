import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import joblib
import spacy
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data = train_data.fillna({'keyword': 'unknown'}) 

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text) 
    text = re.sub(r'\d+', '', text)  
    text = text.lower() 
    text = word_tokenize(text) 
    text = [word for word in text if word.isalpha()] 
    text = [word for word in text if word not in stopwords.words('english')] 
    return ' '.join(text)


train_data['tweet'] = train_data['tweet'].apply(preprocess_text)
test_data['tweet'] = test_data['tweet'].apply(preprocess_text)


X_train = train_data['tweet']
y_train_disaster = train_data['keyword'].str.contains("disaster").astype(int)  # Binary classification

pipeline_disaster = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=200))
])

pipeline_disaster.fit(X_train, y_train_disaster)

joblib.dump(pipeline_disaster, 'disaster_occurrence_model.pkl')


X_test = test_data['tweet']
predictions = pipeline_disaster.predict(X_test)

nlp = spacy.load('en_core_web_sm')

def extract_locations(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'GPE'] 
def get_predicted_places(row):
    if "disaster" in row['keyword']:
        return extract_locations(row['tweet'])
    return []


test_data['predicted_places'] = test_data['tweet'].apply(extract_locations)
all_predicted_places = [place for places in test_data['predicted_places'] for place in places]

location_counts = Counter(all_predicted_places)
top_5_locations = location_counts.most_common(5)

print("Top 5 Predicted Locations:")
for location, count in top_5_locations:
    print(f"{location}: {count}")