import os
import logging
import requests
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set API Key for DeepSeek
DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"  # Replace with your actual API key
DEEPSEEK_API_URL = "https://api.deepseek.com/"

# Load and preprocess documents
def load_and_store_documents():
    url = "https://brainlox.com/courses/category/technical"
    logging.info(f"Loading data from {url}")

    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    logging.info(f"Loaded {len(docs)} document chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    vector_store.persist()
    return vector_store

# Load stored vector database
vector_store = load_and_store_documents()

# Flask setup
app = Flask(__name__)
api = Api(app)

# Query DeepSeek API
def query_deepseek(prompt):
    """Send a request to DeepSeek's API for chat completion."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",  # Update with the correct model name
        "messages": [{"role": "system", "content": "You are a helpful AI assistant."},
                     {"role": "user", "content": prompt}]
    }
    
    response = requests.post(DEEPSEEK_API_URL, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        logging.error(f"DeepSeek API Error: {response.text}")
        return f"Error: {response.text}"

# Chat API Endpoint
class ChatbotAPI(Resource):
    def post(self):
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "Message field is required"}), 400

        logging.info(f"User input: {user_input}")

        # Retrieve documents from the vector store
        retrieved_docs = vector_store.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Format input with retrieved knowledge
        prompt = f"Context:\n{context}\n\nUser Query: {user_input}"

        # Query DeepSeek AI
        response = query_deepseek(prompt)

        return jsonify({"response": response})

api.add_resource(ChatbotAPI, "/chat")

if __name__ == "__main__":
    logging.info("Starting Flask server...")
    app.run(debug=True)