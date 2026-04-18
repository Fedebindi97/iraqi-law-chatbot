# Set up your imports and your flask app.
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from chatbot import *
import os
from dotenv import load_dotenv
from google.oauth2 import service_account

# Load env vars
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
raw_gcp_key = os.getenv("GCP_PRIVATE_KEY", "")
clean_gcp_key = raw_gcp_key.strip().strip('"').strip("'").replace('\\n', '\n')
service_account_info = {
    "project_id": GCP_PROJECT_ID,
    "client_email": os.getenv("GCP_CLIENT_EMAIL"),
    "private_key": clean_gcp_key, 
    "type": "service_account",
    "token_uri": "https://oauth2.googleapis.com/token",
}
scopes = ["https://www.googleapis.com/auth/cloud-platform"]
GCP_CREDENTIALS = service_account.Credentials.from_service_account_info(
    service_account_info, 
    scopes=scopes
)

# Chatbot and app initialization
app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY")
chatbot = Chatbot(
    gcp_credentials=GCP_CREDENTIALS,
    gemini_api_key=GEMINI_API_KEY,
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY,
    gcp_project_id=GCP_PROJECT_ID
)


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
        chat_context_uncurated = chatbot.retrieve_law_chunks(user_input,
                                           collection = "iraqi_laws_en_uncurated_extended",
                                           n_docs=4) # we only retrieve context once when history is empty
        chat_context_curated = chatbot.retrieve_law_chunks(user_input,
                                           collection = "iraqi_laws_en_curated",
                                           n_docs=2) # we retrieve less docs from the curated corpus (as it's smaller)
        chat_context = chat_context_uncurated + "\n\n" + chat_context_curated
    
    bot_answer = chatbot.respond(user_input,
                                  chat_context,
                                  history,
                                  session.get('language','en'))

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
    app.run(debug=True)
