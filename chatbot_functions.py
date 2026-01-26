import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import re

def retrieve_law_chunks(user_question, embed_model, collection):
    # STEP A: Embed the question
    query_vector = embed_model.encode(user_question).tolist()
    
    # STEP B: Retrieve relevant law chunks
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=4  # Top 4 most relevant articles
    )
    
    return "\n\n".join(results['documents'][0])

def get_llm_response(user_query, retrieved_context, chat_history):

    system_instruction = {
        "role": "system", 
        "content": '''You are an expert in Iraqi legislation. Use the provided CONTEXT to answer the QUESTION.
            If the answer is not in the context, say you don't know based on the provided documents.'''
    }

    # 2. Format the Current Question with its specific Law Context
    current_prompt = {
        "role": "user",
        "content": f'''
        
            ******************************************************************
            
            QUESTION:
            {user_query}

            ******************************************************************

            CONTEXT:
            {retrieved_context}

            ******************************************************************'''
    }

    # 3. Build the payload: System + History + Current Prompt
    # We do NOT save 'retrieved_law_text' into the session history!
    full_messages = [system_instruction] + chat_history + [current_prompt]

    # 4. Call Ollama
    response = ollama.chat(model='qwen2.5:7b', messages=full_messages)
    
    return response['message']['content']