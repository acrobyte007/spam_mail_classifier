import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Ensure stopwords and punkt are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load the Bernoulli Naive Bayes model from the pickle file
with open('bernoulli_model.pkl', 'rb') as f:
    bnb = pickle.load(f)

# Load the CountVectorizer from the pickle file
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

# Initialize the Porter Stemmer
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Streamlit app
st.title('SMS and Email spam Classification')

# Input text box
input_text = st.text_area("Enter text for classification:")

# Button to make prediction
if st.button('Classify'):
    # Preprocess the input text
    transformed_text = transform_text(input_text)

    # Convert the text to a document-term matrix
    vectorized_text = cv.transform([transformed_text]).toarray()

    # Make prediction
    prediction = bnb.predict(vectorized_text)

    # Display the prediction
    if prediction[0] == 0:
        st.write('Prediction: Not Spam')
    else:
        st.write('Prediction: Spam')
