# This is the web server (backend) for the Fake News Detector.
# It loads the model trained in 01_eda.ipynb and exposes a /predict
# endpoint that templates/index.html calls when you click "Run AI Analysis".
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)  # make sure the stopword list is available

app = Flask(__name__)
CORS(app)  # allow the frontend to call this API even if served from a different origin

# Load the trained model. We never actually built a combined "pipeline.pkl"
# in the notebook, so this always falls into the except branch and loads
# the separate best_model.pkl (the classifier) + the tfidf vectorizer that
# was saved alongside the train/test data in data.pkl.
try:
    with open('models/pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    use_pipeline = True
    print("Loaded pipeline model")
except FileNotFoundError:
    use_pipeline = False
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/data.pkl', 'rb') as f:
        _, _, _, _, tfidf = pickle.load(f)
    print("Loaded separate model + tfidf")

stop_words = set(stopwords.words('english'))

def clean(text):
    # Must mirror clean_text() in the notebook exactly — the model was
    # trained on text cleaned this way, so predictions on differently
    # cleaned text would be unreliable.
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return ' '.join([w for w in text.split() if w not in stop_words])

@app.route('/')
def home():
    # Serves the single-page frontend (templates/index.html)
    return render_template('index.html')

@app.route('/health')
def health():
    # Simple endpoint to check the server is up (useful for deployment checks)
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Reject empty/missing input before doing any real work
    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({'error': 'No text provided'}), 400

    text = clean(data['text'])

    # If cleaning removed every word (e.g. input was just numbers/punctuation),
    # there's nothing meaningful left for the model to classify
    if not text.strip():
        return jsonify({'error': 'Text has no meaningful words after cleaning'}), 400

    if use_pipeline:
        proba = pipeline.predict_proba([data['text']])[0]
    else:
        vec = tfidf.transform([text])         # turn cleaned text into the same TF-IDF features used in training
        proba = model.predict_proba(vec)[0]   # [P(real), P(fake)]

    # NOTE: this uses a 0.75 confidence threshold instead of the standard 0.5 —
    # the model only labels something FAKE if it's at least 75% sure, otherwise
    # it defaults to REAL. This deliberately biases the app toward fewer false
    # "fake" accusations at the cost of letting more borderline fakes through.
    pred = 1 if proba[1] > 0.75 else 0
    prob = max(proba) * 100
    label = "FAKE" if pred == 1 else "REAL"

    return jsonify({
        'label': label,
        'confidence': round(prob, 1)
    })

if __name__ == '__main__':
    app.run(debug=True)
