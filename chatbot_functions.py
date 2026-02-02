def retrieve_law_chunks(user_question,
                        gemini_client,
                        qdrant_client,
                        collection):
    # STEP A: Use Google's API to get the embedding
    # This replaces 'embed_model.encode'
    result = gemini_client.models.embed_content(
        model="text-embedding-004",
        contents=user_question
    )
    query_vector = result.embeddings[0].values
    
    # STEP B: Query Qdrant (Same as before)
    results = qdrant_client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=4
    ).points
    return "\n\n".join([point.payload['page_content'] for point in results])

def get_llm_response(user_query,
                     retrieved_context,
                     chat_history,
                     gemini_client):

    system_instruction = '''
        You are an expert in Iraqi legislation.
        Use the provided CONTEXT to answer the QUESTION.
        If the answer is not in the context,
        say you don't know based on the provided documents.
    '''

    current_prompt = f'''
        ******************************************************************
        
        QUESTION:
        {user_query}

        ******************************************************************

        CONTEXT:
        {retrieved_context}

        ******************************************************************
    '''

    # build the payload for Gemini
    gemini_history = []
    for msg in chat_history:
        gemini_history.append({"role": msg["role"],
                            "parts": [{"text": msg["content"]}]})
    full_messages = gemini_history + [{"role": "user", "parts": [{"text": current_prompt}]}]

    # 4. Call Gemini
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_messages,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.5, # Lower temperature = more factual for law
        }
    )
    
    return response.text