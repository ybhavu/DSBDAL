from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    result = model(text)[0]
    if result['label'] == 'POSITIVE':
        prediction = 'Positive 😊'
    else:
        prediction = 'Negative 😞'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
