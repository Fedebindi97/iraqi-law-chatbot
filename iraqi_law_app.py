# Set up your imports and your flask app.
from flask import Flask, render_template, request, jsonify, session, redirect
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import re
from chatbot_functions import *

app = Flask(__name__)
app.secret_key = "super_secret_iraqi_law_key"
client = chromadb.PersistentClient(path="./law_database")
embed_model = SentenceTransformer('BAAI/bge-m3', device='mps') # use 'cpu' if no Mac GPU
collection = client.get_collection(name="iraq_laws_en") # iraqi_laws_ar

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/reset', methods=['POST'])
def reset_chat():
    # Clear all session data (history)
    session.clear()
    # Return a success status to the JavaScript caller
    return jsonify({"status": "success", "message": "Chat history cleared."})

@app.route('/ask', methods=['POST'])
def ask():

    history = session.get('history', [])
    user_input = request.json.get('message')
    if history == []:
        chat_context = retrieve_law_chunks(user_input, embed_model, collection) # we only retrieve context once when history is empty
    
    bot_answer = get_llm_response(user_input, chat_context, history)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": bot_answer})
    session['history'] = history[-6:]

    session.modified = True
    
    return jsonify({
        'status': 'success',
        'answer': bot_answer
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
