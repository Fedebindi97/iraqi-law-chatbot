import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import re

client = chromadb.PersistentClient(path="./law_database")
embed_model = SentenceTransformer('BAAI/bge-m3', device='mps') # use 'cpu' if no Mac GPU

def is_arabic(user_question):
    # This pattern matches any string containing Arabic characters
    arabic_pattern = re.compile('[\u0600-\u06FF]+')
    return bool(arabic_pattern.search(user_question))

def query_chatbot(user_question, collection):
    # STEP A: Embed the question
    query_vector = embed_model.encode(user_question).tolist()
    
    # STEP B: Retrieve relevant law chunks
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3  # Top 3 most relevant articles
    )
    
    context = "\n\n".join(results['documents'][0])
    
    # STEP C: Construct the Prompt
    prompt = f"""
    You are an expert in Iraqi legislation. Use the following excerpts from Iraqi law to answer the user's question.
    If the answer is not in the context, say you don't know based on the provided documents.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {user_question}
    
    ANSWER (Direct and citing Articles if possible):
    """
    
    # STEP D: Generate response using Ollama
    response = ollama.chat(model='qwen2.5:7b', messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    return response['message']['content'], results['metadatas'][0]

def chatbot_answers(user_question):
    if is_arabic(user_question):
        collection = client.get_collection(name="iraq_laws_ar")
    else:
        collection = client.get_collection(name="iraq_laws_en")
    answer, sources = query_chatbot(user_question, collection)
    return answer, sources