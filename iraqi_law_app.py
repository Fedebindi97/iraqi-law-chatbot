# Set up your imports and your flask app.
from flask import Flask, render_template, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import re

from chatbot_functions import chatbot_answers

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('message')
    
    # Simulate or call your chatbot logic here
    # answer, sources = query_chatbot(user_input)
    answer, sources = chatbot_answers(user_input)
    
    return jsonify({
        'status': 'success',
        'answer': answer
    })

@app.route('/info_project')
def info_project():

    # Return the information to the report page html.
    return render_template('info_me.html')


# This page will be the page after the form
@app.route('/info_me')
def info_me():

    # Return the information to the report page html.
    return render_template('info_me.html')

if __name__ == '__main__':
    app.run(debug=True)
