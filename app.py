from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load the data
with open('train.txt', 'r') as file:
    data = file.readlines()

# Split data into sentences and labels
sentences, labels = zip(*[line.strip().split(';') for line in data])

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Transform sentences into numerical vectors
X = vectorizer.fit_transform(sentences)

# Initialize Multinomial Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(X, labels)

# Function to predict sentiment of user input
def predict_sentiment(input_sentence):
    input_vector = vectorizer.transform([input_sentence])
    prediction = classifier.predict(input_vector)
    probabilities = classifier.predict_proba(input_vector)
    sentiment = prediction[0]
    positivity = probabilities[0][list(classifier.classes_).index('joy')]
    negativity = 1 - positivity
    return sentiment, positivity * 100, negativity * 100

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    positivity = None
    negativity = None
    if request.method == 'POST':
        input_sentence = request.form['input_sentence']
        sentiment, positivity, negativity = predict_sentiment(input_sentence)
    return render_template('index.html', sentiment=sentiment, positivity=positivity, negativity=negativity)

if __name__ == '__main__':
    app.run(debug=True)
