from flask import Flask, request, jsonify
import openai
from supabase import create_client
import os
import numpy as np
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize clients
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text):
    """Generates an embedding for the given text using OpenAI API."""
    try:
        response = openai_client.embeddings.create(input=[text], model="text-embedding-ada-002")
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def fetch_relevant_context(question):
    """Fetches the most relevant content chunk from Supabase using vector similarity."""
    embedding = get_embedding(question)
    if embedding is None:
        return ""

    # Fetch all content chunks from Supabase
    response = supabase_client.table("embeddings").select("content, embedding").execute()
    chunks = response.data or []

    best_match, best_score = None, -1

    for chunk in chunks:
        chunk_embedding = chunk.get("embedding")
        if not chunk_embedding:
            continue  # Skip if missing

        try:
            chunk_embedding = np.array(json.loads(chunk_embedding), dtype=np.float32)  # Convert JSON string to array
        except (ValueError, TypeError, json.JSONDecodeError):
            continue  # Skip invalid embeddings

        # Compute cosine similarity
        if chunk_embedding.any():
            score = np.dot(embedding, chunk_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(chunk_embedding))
            if score > best_score:
                best_score = score
                best_match = chunk["content"]

    return best_match or ""

@app.route("/ask", methods=["POST"])
def ask_question():
    """Handles incoming user questions and returns AI-generated responses."""
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    context = fetch_relevant_context(question)

    if not context:
        return jsonify({"answer": "I couldn't find relevant information."})

    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        answer = response.choices[0].message.content.strip() if response.choices else "I couldn't generate an answer based on the given context."
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": "Failed to generate an answer."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not set
    app.run(host="0.0.0.0", port=port)


