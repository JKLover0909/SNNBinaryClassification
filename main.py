from flask import Flask, jsonify, render_template, request
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import standard_data

import model_tensorflow

print("Đọc dữ liệu")
data_en = pd.read_csv('data_en.csv')
data_vi = pd.read_csv('data_vi.csv', encoding="utf-8")
max_len = 100


model_en, tokenizer_en = model_tensorflow.process(data_en, "en")

model_vi, tokenizer_vi = model_tensorflow.process(data_vi, "vi")

# app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/process_mail', methods=['POST'])
def process_mail():
    result = None
    text = request.form.get('text')
    language = request.form.get('language')
    result = process_text(text, language)

    return jsonify({'result': result})


def process_text(text, language):
    
    # Tokenize và đệm văn bản
    if language == "en":
        text = standard_data.standard_en(text)
        input_data = tokenizer_en.texts_to_sequences([text])
        input_data = pad_sequences(input_data, maxlen = max_len, padding = 'post', truncating = 'post')
        prediction = model_en.predict(input_data)
    else:
        text = standard_data.standard_vi(text)
        input_data = tokenizer_vi.texts_to_sequences([text])
        input_data = pad_sequences(input_data, maxlen = max_len, padding = 'post', truncating = 'post')
        prediction = model_vi.predict(input_data)
    
    # Dự đoán
    if prediction[0] > 0.5:
        return "Spam mail"
    else:
        return "Ham mail"

if __name__ == "__main__":
    app.run()