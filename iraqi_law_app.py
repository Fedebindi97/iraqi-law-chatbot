# Set up your imports and your flask app.
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from chatbot_functions import *
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY")
embed_model = SentenceTransformer('BAAI/bge-m3', device='mps') # use 'cpu' if no Mac GPU

qdrant_client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


@app.route('/')
def index():
    # Clear chatbot variables at the start of a new session
    session['history'] = []
    session['chat_context'] = ''
    session.modified = True
    return render_template('chatbot.html', language=session.get('language', 'en'))

@app.route('/reset', methods=['POST'])
def reset_chat():
    # Clear chatbot variables at the start of a new session
    session['history'] = []
    session['chat_context'] = ''
    session.modified = True
    # Return a success status to the JavaScript caller
    return jsonify({"status": "success", "message": "Chat history cleared."})

@app.route('/set_language/<lang>')
def set_language(lang):
    if lang in ['en', 'ar']:
        session['language'] = lang
    
    # Redirect back to the previous page, or home if referrer is missing
    return redirect(request.referrer or url_for('index'))

@app.route('/ask', methods=['POST'])
def ask():

    history = session.get('history', [])
    chat_context = session.get('chat_context', '')
    user_input = request.json.get('message')
    if chat_context == '':
        chat_context = retrieve_law_chunks(user_input,
                                           embed_model,
                                           qdrant_client,
                                           collection = "iraqi_laws_en" if session.get('language', 'en') == 'en' else "iraqi_laws_ar") # we only retrieve context once when history is empty
    
    bot_answer = get_llm_response(user_input, chat_context, history, gemini_client)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "model", "content": bot_answer})
    session['history'] = history[-6:]
    session['chat_context'] = chat_context
    session.modified = True
    
    return jsonify({
        'status': 'success',
        'answer': bot_answer
    })

@app.route('/info_project')
def info_project():

    # Return the information to the report page html.
    return render_template('info_project.html', language=session.get('language', 'en'))


# This page will be the page after the form
@app.route('/info_me')
def info_me():

    # Return the information to the report page html.
    return render_template('info_me.html', language=session.get('language', 'en'))

if __name__ == '__main__':
    app.run()
