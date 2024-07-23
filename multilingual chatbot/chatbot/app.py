from flask import Flask, request, jsonify, render_template
import joblib
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/chatbot_model.pkl')

# Load mBART model and tokenizer
mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Map languages to mBART language codes
language_code_map = {
    'en': 'en_XX',
    'hi': 'hi_IN',
    'ml': 'ml_IN',
    'ta': 'ta_IN',
    'gu': 'gu_IN'
}

def translate_text(text, source_lang, target_lang):
    tokenizer = mbart_tokenizer
    model = mbart_model

    tokenizer.src_lang = language_code_map[source_lang]
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[language_code_map[target_lang]])
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['message']
    user_language = request.form['language']

    # Translate the user message to English
    if user_language != 'en':
        translated_message = translate_text(user_message, source_lang=user_language, target_lang='en')
    else:
        translated_message = user_message

    # Get the response from the model
    response = model.predict([translated_message])[0]

    # Translate the response back to the user's language
    if user_language != 'en':
        translated_response = translate_text(response, source_lang='en', target_lang=user_language)
    else:
        translated_response = response

    return jsonify({'response': translated_response})

if __name__ == "__main__":
    app.run(debug=True)
