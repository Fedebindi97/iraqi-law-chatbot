from google.cloud import discoveryengine_v1alpha as discoveryengine
from qdrant_client import QdrantClient
from google import genai
from system_instructions import *

class Chatbot:

    def __init__(self,
                 gcp_credentials,
                 qdrant_url,
                 qdrant_api_key,
                 gemini_api_key,
                 gcp_project_id):
        self.rank_client = discoveryengine.RankServiceClient(
            credentials=gcp_credentials
        )
        self.qdrant_client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key,
        )
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self.gcp_project_id = gcp_project_id

    def retrieve_law_chunks(self,
                        user_query,
                        collection,
                        n_docs):

        result = self.gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=user_query,
            config={'task_type': 'RETRIEVAL_QUERY', 'output_dimensionality': 768}
        )
        query_vector = result.embeddings[0].values

        initial_results = self.qdrant_client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=20  # Increased limit for the reranker pool
        ).points

        # Extract the text chunks for reranking
        documents = [point.payload['page_content'] for point in initial_results]

        ranking_config = self.rank_client.ranking_config_path(
            project=self.gcp_project_id,
            location="global",
            ranking_config="default_ranking_config",
        )

        # Prepare records for the API
        records = [
            discoveryengine.RankingRecord(id=str(i), content=doc) 
            for i, doc in enumerate(documents)
        ]

        rank_request = discoveryengine.RankRequest(
            ranking_config=ranking_config,
            model="semantic-ranker-512@latest",
            query=user_query,
            records=records,
        )

        # Execute Reranking
        response = self.rank_client.rank(request=rank_request)

        # We take the top 4 docs
        top_chunks = [r.content for r in response.records[:n_docs]]
        
        return "\n\n".join(top_chunks)
    
    def respond(self,
                     user_query,
                     retrieved_context,
                     chat_history,
                     language):

        if language == 'en':
            system_instructions = SYSTEM_INSTRUCTIONS_EN
            current_prompt = f'''
                ******************************************************************
                
                QUESTION:
                {user_query}

                ******************************************************************

                CONTEXT:
                {retrieved_context}

                ******************************************************************
            '''
        else:
            system_instructions = SYSTEM_INSTRUCTIONS_AR
            current_prompt = f'''
                ******************************************************************
                
                السؤال:
                {user_query}

                ******************************************************************

                السياق:
                {retrieved_context}

                ******************************************************************
            '''

        # build the payload for Gemini
        gemini_history = []
        for msg in chat_history:
            gemini_history.append({"role": msg["role"],
                                "parts": [{"text": msg["content"]}]})
        full_messages = gemini_history + [{"role": "user", "parts": [{"text": current_prompt}]}]

        # call Gemini
        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=full_messages,
            config={
                "system_instruction": system_instructions,
                "temperature": 0.5 # Lower temperature = more factual for law
            }
        )
        
        return response.text