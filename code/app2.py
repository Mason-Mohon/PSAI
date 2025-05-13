from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from pathlib import Path
import qdrant_client
from qdrant_client.models import Distance
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import json

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv("/Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/code/.env")

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-001"

# Initialize clients
qdrant_client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

genai_client = genai.Client(api_key=GOOGLE_API_KEY)

# New function to refine the query
def refine_query_for_semantic_search(query_text: str, client: genai.Client, model_name: str) -> str:
    prompt = f"""Rewrite the following user query to be optimized for semantic search against a knowledge base primarily focused on Phyllis Schlafly's life, work, and conservative viewpoints.
Extract the key entities, topics, and the core intent. Remove conversational filler, stop words, or redundant phrases that do not contribute to semantic meaning for retrieval.
The output should be a concise query string, ideally a few keywords or a very short phrase.

User Query: "{query_text}"
Optimized Search Query:"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for more deterministic and focused output
                max_output_tokens=60, # Refined query should be concise
                stop_sequences=["\n"] # Stop if it starts generating new lines
            )
        )
        refined_query = response.text.strip()
        # Clean up potential prefixes if the model includes them despite the prompt
        if refined_query.startswith("Optimized Search Query:"):
            refined_query = refined_query.replace("Optimized Search Query:", "").strip()
        refined_query = refined_query.strip('"') # Remove potential surrounding quotes
        
        print(f"Original query for refinement: '{query_text}'")
        print(f"Refined query: '{refined_query}'")
        
        # Fallback to original query if refinement results in an empty string
        return refined_query if refined_query else query_text
    except Exception as e:
        print(f"Error refining query: {e}")
        return query_text # Fallback to original query on error

# Get available collections
def get_available_collections():
    collections = [c.name for c in qdrant_client.get_collections().collections]
    return collections

# Perform semantic search
def semantic_search(query_text, collections, limit=5, similarity_threshold=0.0):
    query_vector = model.encode(query_text).tolist()
    all_results = []
    for collection_name in collections:
        try:
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                score_threshold=similarity_threshold
            )
            for idx, result in enumerate(search_results):
                formatted_result = {
                    "collection": collection_name,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": {},
                    "ref_id": f"REF_{idx+1}"
                }
                if "metadata" in result.payload:
                    formatted_result["metadata"] = result.payload["metadata"]
                else:
                    for key in result.payload:
                        if key != "text":
                            formatted_result["metadata"][key] = result.payload[key]
                all_results.append(formatted_result)
        except Exception as e:
            print(f"Error searching collection {collection_name}: {e}")
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:limit]

# Generate response using Google Gemini
def generate_gemini_response(query, context_chunks, temperature=0.7):
    try:
        formatted_chunks = []
        for idx, chunk in enumerate(context_chunks):
            metadata = chunk.get('metadata', {})
            author = metadata.get('author', 'Unknown')
            book_title = metadata.get('book_title', '')
            publication_year = metadata.get('publication_year', '')
            doc_type = metadata.get('doc_type', '')
            source_file = metadata.get('source_file', '')

            source_info = f"Collection: {chunk['collection']}"
            if book_title:
                source_info += f", Book: {book_title}"
            if publication_year:
                source_info += f", Year: {publication_year}"
            if author:
                source_info += f", Author: {author}"
            if doc_type:
                source_info += f", Type: {doc_type}"
            if source_file:
                source_info += f", File: {source_file}"

            formatted_chunks.append(f"[REF_{idx+1}] Source [{source_info}]: {chunk['text']}")

        formatted_context = "\n\n".join(formatted_chunks)

        system_instruction = (
            "You are playing the role of Phyllis Schlafly answering questions based solely on the provided context. "
            "If the context lacks sufficient detail to answer, say so. Use endnotes for citations but omit source filenames. "
            "If the source is your own writing, speak in your voice as if it is your own. Present a confident, conservative tone."
        )

        prompt = (
            f"Context:\n{formatted_context}\n\n"
            f"Question: {query}\n\n"
            "Answer the question strictly based on the above context. "
            "When referencing any chunk, indicate its reference ID inline (e.g., [REF_1], [REF_2]) and include numbered endnotes at the end of your response. "
            "Each endnote should follow this format: [n] Title of piece, publication (e.g., Phyllis Schlafly Report or book title), date, author. "
            "If the information is not visible in the chunk itself, use the information within the metadata to generate the citation."
            "Do not include source filenames. If the author is Phyllis Schlafly, treat it as your own words and omit the author from the endnote."
        )

        token_response = genai_client.models.count_tokens(
            model=GEMINI_MODEL,
            contents=prompt
        )
        input_token_count = token_response.total_tokens

        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
                max_output_tokens=1024,
            ),
        )

        output_token_response = genai_client.models.count_tokens(
            model=GEMINI_MODEL,
            contents=response.text
        )
        output_token_count = output_token_response.total_tokens

        return {
            "text": response.text,
            "token_info": {
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count
            }
        }
    except Exception as e:
        return {"text": f"Error generating response: {str(e)}", "token_info": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}

@app.route('/')
def index():
    collections = get_available_collections()
    return render_template('index.html', collections=collections)

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    original_query_text = data.get('query', '')
    selected_collections = data.get('collections', [])
    chunk_limit = int(data.get('chunk_limit', 5))
    temperature = float(data.get('temperature', 0.7))
    similarity_threshold = float(data.get('similarity_threshold', 0.0))

    if not original_query_text:
        return jsonify({"error": "Query is required"}), 400

    if not selected_collections:
        return jsonify({"error": "At least one collection must be selected"}), 400

    # Refine the query before semantic search
    refined_query_text = refine_query_for_semantic_search(original_query_text, genai_client, GEMINI_MODEL)

    search_results = semantic_search(
        refined_query_text,  # Use the refined query for searching
        selected_collections,
        limit=chunk_limit,
        similarity_threshold=similarity_threshold
    )

    # Use the original query text for generating the final response, as that's what the user asked
    gemini_response = generate_gemini_response(original_query_text, search_results, temperature)

    return jsonify({
        "original_query": original_query_text,
        "refined_query_for_search": refined_query_text,
        "query_used_for_search": refined_query_text, # Explicitly state what was used
        "chunks": search_results,
        "response": gemini_response.get("text", "Error generating response"),
        "token_info": gemini_response.get("token_info", {})
    })

if __name__ == '__main__':
    if not GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY is not set. Please add it to your .env file.")

    try:
        collections = get_available_collections()
        print(f"Available collections: {collections}")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")

    app.run(debug=True)

#To run:
#/Users/mason/opt/anaconda3/envs/psai/bin/python /Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/code/app2.py