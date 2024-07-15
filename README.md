
# Disaster Occurrence Prediction from Tweets

## Overview

This project focuses on predicting whether a tweet is related to a disaster or not. We'll preprocess the tweet text, train a binary classifier, and extract location information from relevant tweets.

## Prerequisites

- Python 3.x
- Libraries: pandas, nltk, scikit-learn, spacy

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/disaster-prediction.git
   cd disaster-prediction
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your dataset**:
   - Ensure you have labeled tweets (disaster-related or not) in CSV format.
   - Replace `train.csv` and `test.csv` with your data.

2. **Preprocess the text**:
   - Clean the tweet text using the provided `preprocess_text` function.
   - Tokenize, remove stopwords, and convert to lowercase.

3. **Train the model**:
   - Use the `TfidfVectorizer` and `LogisticRegression` for binary classification.
   - Save the trained model using `joblib`.

4. **Extract locations**:
   - If a tweet is related to a disaster, extract locations using spaCy.


