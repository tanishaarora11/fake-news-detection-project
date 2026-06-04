from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model and tfidf
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/data.pkl', 'rb') as f:
    _, _, _, _, tfidf = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean(text):
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return ' '.join([w for w in text.split() if w not in stop_words])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = clean(data['text'])
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    prob = max(model.predict_proba(vec)[0]) * 100
    label = "FAKE" if pred == 1 else "REAL"
    return jsonify({'label': label, 'confidence': round(prob, 1)})

if __name__ == '__main__':
    app.run(debug=True)

