# Chat-bot-Sun-C
This project is a multilingual chatbot built using Flask, transformers, and scikit-learn. The chatbot supports multiple languages and can respond to user queries in the selected language.

Table of Contents
Features
Installation
Usage
Folder Structure
License
Features
Multilingual support with translation between English and several Indian languages.
Flask web interface for interaction.
Uses an mBART model for translation and a custom-trained model for responses.
Installation
To set up the project on your local machine, follow these steps:

Clone the Repository:

bash
Copy code
git clone <repository-URL>
cd <repository-folder>
Create and Activate a Virtual Environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:

Make sure you have pip installed, and then install the required Python libraries:

bash
Copy code
pip install -r requirements.txt
Download Pre-trained Models:

Ensure you have internet access to download the pre-trained mBART model and tokenizer. These will be cached by the transformers library.

Prepare Data and Train the Model:

Place your question-answer data in data/qa.csv. The file should have two columns: question and answer.

Run the training script to train and save the model:

bash
Copy code
python train_model.py
Usage
Start the Flask Application:

bash
Copy code
python app.py
Open Your Browser:

Navigate to http://127.0.0.1:5000/ to access the chatbot interface.

Interact with the Chatbot:

Enter your message and select your language.
Click "Send" to get a response from the chatbot.
Folder Structure
app.py: The main Flask application script.
train_model.py: Script for training and saving the chatbot model.
data/qa.csv: Data file used for training the model (place your own data here).
model/chatbot_model.pkl: Saved model file after training.
templates/index.html: HTML template for the chatbot interface.
requirements.txt: List of Python dependencies.
requirements.txt
Here is the requirements.txt file with all necessary libraries:

makefile
Copy code
Flask==2.0.2
joblib==1.2.0
pandas==1.5.0
scikit-learn==1.2.0
torch==2.0.1
transformers==4.30.0
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Feel free to modify the README.md to better suit your project or add more information as needed.
