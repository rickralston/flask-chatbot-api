from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import openai
from supabase import create_client
import os
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize OpenAI and Supabase clients
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_relevant_context(question):
    """Fetches the most relevant content chunk from Supabase."""
    embedding = get_embedding(question)
    response = supabase_client.table("embeddings").select("content, embedding").execute()
    
    chunks = response.data
    if not chunks:
        return ""
    
    best_match, best_score = None, -1
    
    for chunk in chunks:
        chunk_embedding = chunk.get("embedding")
        if not chunk_embedding:
            continue

        try:
            chunk_embedding = np.array(json.loads(chunk_embedding), dtype=np.float32)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue

        if chunk_embedding.any():
            score = np.dot(embedding, chunk_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(chunk_embedding))
            if score > best_score:
                best_score = score
                best_match = chunk["content"]
    
    return best_match if best_match else ""

def get_embedding(text):
    """Generates an embedding for the given text."""
    response = openai_client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return np.array(response.data[0].embedding, dtype=np.float32)

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    
    context = fetch_relevant_context(question)
    
    if not context:
        return jsonify({"answer": "I couldn't find relevant information in my knowledge base."})
    
    system_prompt = """
    You are an AI assistant that answers questions using ONLY the provided context.
    Do not use outside knowledge, even if the answer seems obvious.
    Your tone should match the style of the given content—clear, engaging, and approachable.
    Keep answers conversational, direct, and to the point. Avoid sounding overly formal.
    If the context includes a definition (e.g., 'X stands for Y'), prioritize extracting that.
    If the answer is in the context, provide it naturally, like you’re chatting with a friend or colleague.
    If the context does not fully answer the question, say so instead of guessing.
    If a term is defined in the context, explain it simply and clearly.
    Keep answers concise, aiming for **2-4 sentences max**. 
    Avoid long-winded explanations or unnecessary details.
    If the answer isn’t fully available, be honest about what’s missing instead of guessing.
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
        ],
        temperature=0.2,  # Lowered temperature for more controlled responses
    )
    
    answer = response.choices[0].message.content.strip() if response.choices else "I couldn't generate an answer based on the given context."
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
