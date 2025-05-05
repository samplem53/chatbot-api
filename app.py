import os
import requests
import warnings
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

# Setup
load_dotenv(override=True)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Flask app
app = Flask(__name__)

# Custom retriever
class SupabaseRetriever(BaseRetriever):
    url: str = f"{SUPABASE_URL}/rest/v1/rpc/match_documents"
    headers: dict = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json"
    }
    embedding_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embedding_model.embed_query(query)
        payload = {
            "query_embedding": query_embedding,
            "match_count": 3,
            "filter": {}
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        results = response.json()

        documents = []
        for result in results:
            documents.append(
                Document(
                    page_content=result["content"],
                    metadata=result["metadata"]
                )
            )
        return documents

# Load LLM
def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": "512"
        }
    )

# Custom prompt
CUSTOM_PROMPT_TEMPLATE = """
You are a highly knowledgeable and professional Indian Legal Assistant.
Use the provided context to accurately answer the user's legal question.

Always answer strictly based on Indian law.
If you don't know the answer from the context, clearly state you don't know.

Context: {context}
Question: {question}

Please provide a formal and precise legal answer.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Create QA pipeline
retriever = SupabaseRetriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Flask route
@app.route("/query", methods=["POST"])
def legal_assistant():
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    try:
        response = qa_chain.invoke({"query": user_query})
        return jsonify({
            "question": user_query,
            "answer": response["result"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Entry point
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
