from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

# Try loading as pipeline first, fall back to separate model+tfidf
try:
    with open('models/pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    use_pipeline = True
    print("Loaded pipeline model")
except:
    use_pipeline = False
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/data.pkl', 'rb') as f:
        _, _, _, _, tfidf = pickle.load(f)
    print("Loaded separate model + tfidf")

stop_words = set(stopwords.words('english'))

def clean(text):
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return ' '.join([w for w in text.split() if w not in stop_words])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({'error': 'No text provided'}), 400

    text = clean(data['text'])

    if not text.strip():
        return jsonify({'error': 'Text has no meaningful words after cleaning'}), 400

    if use_pipeline:
        proba = pipeline.predict_proba([data['text']])[0]
    else:
        vec = tfidf.transform([text])
        proba = model.predict_proba(vec)[0]

    pred = 1 if proba[1] > 0.75 else 0
    prob = max(proba) * 100
    label = "FAKE" if pred == 1 else "REAL"

    return jsonify({
        'label': label,
        'confidence': round(prob, 1)
    })

if __name__ == '__main__':
    app.run(debug=True)
