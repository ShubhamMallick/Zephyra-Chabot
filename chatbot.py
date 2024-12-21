import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents and pre-trained model
try:
    intents = json.loads(open(r'C:\Users\aditya\Desktop\ChatBot using Python\chatbot\intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')
except FileNotFoundError as e:
    print(f"Error: Missing file {e.filename}. Ensure all required files are present.")
    exit()

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the user input."""
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Convert user input into a bag-of-words representation."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict the intent of the user input."""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    """Generate a response based on predicted intent."""
    if not intents_list:
        return "Sorry, I didn't understand that. Can you please rephrase?"

    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "Sorry, I couldn't process that request."

# Chatbot loop
print("GO! Bot is running! (Type 'exit' to stop)")

while True:
    message = input("You: ")
    if message.lower() == 'exit':
        print("Bot: Goodbye!")
        break

    try:
        predicted_intents = predict_class(message)
        bot_response = get_response(predicted_intents, intents)
        print(f"Bot: {bot_response}")
    except Exception:
        print("Bot: Sorry, something went wrong.")
