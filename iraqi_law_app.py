# Set up your imports and your flask app.
from flask import Flask, render_template, request, jsonify, session, redirect
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from chatbot_functions import *
from google import genai

GEMINI_API_KEY = "AIzaSyCTlCxD-HwRKUNKMvyJOqVEQtzxmQk6EJk"
QDRANT_URL = "https://fc237fd3-4cd9-4a72-91a4-153b5c17a170.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.cXFgmSC0KFc-lZFB6Kb_YIY252x0a5stGT5R-RWTOY8"

app = Flask(__name__)
app.secret_key = b'_5#yf23543d2LF4Qsgfh5hh58z\n\xec]/'
embed_model = SentenceTransformer('BAAI/bge-m3', device='mps') # use 'cpu' if no Mac GPU

qdrant_client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
)
collection = "iraqi_laws_en"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


@app.route('/')
def index():
    session.clear()
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
    chat_context = session.get('chat_context', '')
    user_input = request.json.get('message')
    if chat_context == '':
        chat_context = retrieve_law_chunks(user_input,
                                           embed_model,
                                           qdrant_client,
                                           collection) # we only retrieve context once when history is empty
    
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
    return render_template('info_project.html')


# This page will be the page after the form
@app.route('/info_me')
def info_me():

    # Return the information to the report page html.
    return render_template('info_me.html')

if __name__ == '__main__':
    app.run(debug=True)
