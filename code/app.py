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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Add this to your .env file
GEMINI_MODEL = "gemini-2.0-flash-001"  # Default model, can be changed if needed

# Initialize clients
qdrant_client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Google Gemini client with the updated SDK method
genai_client = genai.Client(api_key=GOOGLE_API_KEY)


# Get available collections
def get_available_collections():
    collections = [c.name for c in qdrant_client.get_collections().collections]
    return collections

# Perform semantic search
def semantic_search(query_text, collections, limit=5, similarity_threshold=0.0):
    """Perform semantic search across specified collections with similarity threshold."""
    # Embed the query
    query_vector = model.encode(query_text).tolist()
    
    all_results = []
    
    # Search each collection
    for collection_name in collections:
        try:
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                score_threshold=similarity_threshold  # Apply similarity threshold
            )
            
            # Format results with collection source
            for result in search_results:
                formatted_result = {
                    "collection": collection_name,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": {}
                }
                
                # Extract metadata
                if "metadata" in result.payload:
                    formatted_result["metadata"] = result.payload["metadata"]
                else:
                    # If metadata is directly in payload
                    for key in result.payload:
                        if key != "text":
                            formatted_result["metadata"][key] = result.payload[key]
                
                all_results.append(formatted_result)
        
        except Exception as e:
            print(f"Error searching collection {collection_name}: {e}")
    
    # Sort all results by score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top results across all collections
    return all_results[:limit]

# Generate response using Google Gemini
def generate_gemini_response(query, context_chunks, temperature=0.7):
    """Generate a response using Google Gemini API."""
    try:
        # Format context for the prompt with detailed source information
        formatted_chunks = []
        for chunk in context_chunks:
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
                
            formatted_chunks.append(f"Source [{source_info}]: {chunk['text']}")
            
        formatted_context = "\n\n".join(formatted_chunks)
        
        # Create the prompt with system instructions
        system_instruction = (
            "You are playing the role of Phyllis Schlafly that answers questions based on the provided context. "
            "If the context doesn't contain relevant information to answer the question, "
            "say that you don't have enough information. "
            "Cite your sources with endnotes at the end of your response. The endnotes should include the name of the publication, date, author, and title of piece if applicable. "
            "The endnotes should NOT include the source filename. "
            "Cite the author, but if it is Phyllis Schlafly, just say it as your own words because you are roleplaying as Phyllis. "
            "You should use a confident and assertive tone, and present conservative viewpoints consistent with Phyllis Schlafly's ideology. "
            "Respond in the first person as if you are Phyllis Schlafly herself, emulating her writing style and voice as much as possible. "
            "Use the exact text of Phyllis Schlafly's writings when applicable, without quotation, to emulate her voice as much as possible."
        )
        
        # Create the prompt for Gemini - emphasizing the Phyllis Schlafly role
        prompt = f"Context:\n{formatted_context}\n\nQuestion: {query}\n\nRemember that you are Phyllis Schlafly herself. Answer the question based only on the provided context, using your conservative viewpoint and assertive tone. If the source is your own writing, you should acknowledge it as your own words. If the information comes from someone else, you should cite them appropriately."
        
        # Count input tokens
        token_response = genai_client.models.count_tokens(
            model=GEMINI_MODEL,
            contents=prompt
        )
        input_token_count = token_response.total_tokens
        
        # Call Gemini API
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
                max_output_tokens=1024,
            ),
        )
        
        # Count output tokens
        output_token_response = genai_client.models.count_tokens(
            model=GEMINI_MODEL,
            contents=response.text
        )
        output_token_count = output_token_response.total_tokens
        
        # Prepare response with token information
        result = {
            "text": response.text,
            "token_info": {
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count
            }
        }
        
        return result
    except Exception as e:
        return {"text": f"Error generating response: {str(e)}", "token_info": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}

# Routes
@app.route('/')
def index():
    """Render the main page."""
    collections = get_available_collections()
    return render_template('index.html', collections=collections)

@app.route('/api/query', methods=['POST'])
def query():
    """API endpoint for queries."""
    data = request.json
    
    query_text = data.get('query', '')
    selected_collections = data.get('collections', [])
    chunk_limit = int(data.get('chunk_limit', 5))
    temperature = float(data.get('temperature', 0.7))
    similarity_threshold = float(data.get('similarity_threshold', 0.0))
    
    # Validate input
    if not query_text:
        return jsonify({"error": "Query is required"}), 400
    
    if not selected_collections:
        return jsonify({"error": "At least one collection must be selected"}), 400
    
    # Perform search
    search_results = semantic_search(
        query_text, 
        selected_collections, 
        limit=chunk_limit,
        similarity_threshold=similarity_threshold
    )
    
    # Generate response with Gemini
    gemini_response = generate_gemini_response(query_text, search_results, temperature)
    
    # Return results
    return jsonify({
        "query": query_text,
        "chunks": search_results,
        "response": gemini_response.get("text", "Error generating response"),
        "token_info": gemini_response.get("token_info", {})
    })

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create the HTML template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PSAI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
            background-color: #f5f5f5;
        }
        .card {
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .result-card {
            margin-top: 10px;
            border-left: 4px solid #6c757d;
            padding-left: 10px;
        }
        .metadata {
            font-size: 0.8rem;
            color: #6c757d;
        }
        .collection-tag {
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 12px;
            background-color: #e9ecef;
            display: inline-block;
            margin-right: 5px;
        }
        #response-container {
            border-radius: 10px;
            background-color: #fff;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .settings-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .loading {
            text-align: center;
            margin: 20px 0;
            display: none;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mt-4 mb-4 text-center">Phyllis AI Prototype</h1>
        
        <div class="row">
            <div class="col-lg-4">
                <div class="card settings-card">
                    <div class="card-body">
                        <h5 class="card-title">Settings</h5>
                        
                        <form id="queryForm">
                            <div class="mb-3">
                                <label for="collections" class="form-label">Collections</label>
                                <div id="collections-container">
                                    {% for collection in collections %}
                                    <div class="form-check">
                                        <input class="form-check-input collection-checkbox" type="checkbox" value="{{ collection }}" id="collection-{{ collection }}">
                                        <label class="form-check-label" for="collection-{{ collection }}">
                                            {{ collection }}
                                        </label>
                                    </div>
                                    {% endfor %}
                                </div>
                                <div class="mt-2">
                                    <button type="button" id="selectAll" class="btn btn-sm btn-outline-primary">Select All</button>
                                    <button type="button" id="deselectAll" class="btn btn-sm btn-outline-secondary">Deselect All</button>
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <label for="chunkLimit" class="form-label">Number of chunks: <span id="chunkLimitValue">5</span></label>
                                <input type="range" class="form-range" min="1" max="20" step="1" id="chunkLimit" value="5">
                            </div>
                            
                            <div class="mb-3">
                                <label for="temperature" class="form-label">Temperature: <span id="temperatureValue">0.7</span></label>
                                <input type="range" class="form-range" min="0" max="1" step="0.1" id="temperature" value="0.7">
                            </div>
                            
                            <div class="mb-3">
                                <label for="similarityThreshold" class="form-label">Similarity Threshold: <span id="similarityThresholdValue">0.0</span></label>
                                <input type="range" class="form-range" min="0" max="1" step="0.05" id="similarityThreshold" value="0">
                                <small class="text-muted">Higher values return only more relevant chunks (0 = no threshold)</small>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="col-lg-8">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Query</h5>
                        <div class="mb-3">
                            <textarea class="form-control" id="queryInput" rows="3" placeholder="Enter your query here..."></textarea>
                        </div>
                        <div class="d-grid">
                            <button id="submitQuery" class="btn btn-primary">Submit</button>
                        </div>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p>Processing your query...</p>
                </div>
                
                <div id="response-container" style="display: none;">
                    <h5>PS AI Response:</h5>
                    <div id="gemini-response" class="mt-3"></div>
                    
                    <div class="mt-2 p-2 bg-light rounded">
                        <h6>Token Information:</h6>
                        <div id="token-info" class="d-flex">
                            <div class="me-3">
                                <span class="fw-bold">Input:</span> <span id="input-tokens">0</span>
                            </div>
                            <div class="me-3">
                                <span class="fw-bold">Output:</span> <span id="output-tokens">0</span>
                            </div>
                            <div>
                                <span class="fw-bold">Total:</span> <span id="total-tokens">0</span>
                            </div>
                        </div>
                    </div>
                    
                    <hr>
                    
                    <div class="mt-4">
                        <h5>Reference Chunks:</h5>
                        <div id="chunks-container"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Range sliders
            const chunkLimit = document.getElementById('chunkLimit');
            const chunkLimitValue = document.getElementById('chunkLimitValue');
            const temperature = document.getElementById('temperature');
            const temperatureValue = document.getElementById('temperatureValue');
            const similarityThreshold = document.getElementById('similarityThreshold');
            const similarityThresholdValue = document.getElementById('similarityThresholdValue');
            
            chunkLimit.addEventListener('input', function() {
                chunkLimitValue.textContent = this.value;
            });
            
            temperature.addEventListener('input', function() {
                temperatureValue.textContent = this.value;
            });
            
            similarityThreshold.addEventListener('input', function() {
                similarityThresholdValue.textContent = this.value;
            });
            
            // Collection selection buttons
            const selectAll = document.getElementById('selectAll');
            const deselectAll = document.getElementById('deselectAll');
            const collectionCheckboxes = document.querySelectorAll('.collection-checkbox');
            
            selectAll.addEventListener('click', function() {
                collectionCheckboxes.forEach(checkbox => {
                    checkbox.checked = true;
                });
            });
            
            deselectAll.addEventListener('click', function() {
                collectionCheckboxes.forEach(checkbox => {
                    checkbox.checked = false;
                });
            });
            
            // Form submission
            const submitButton = document.getElementById('submitQuery');
            const queryInput = document.getElementById('queryInput');
            const loading = document.getElementById('loading');
            const responseContainer = document.getElementById('response-container');
            const geminiResponse = document.getElementById('gemini-response');
            const chunksContainer = document.getElementById('chunks-container');
            
            submitButton.addEventListener('click', async function() {
                // Get selected collections
                const selectedCollections = [];
                collectionCheckboxes.forEach(checkbox => {
                    if (checkbox.checked) {
                        selectedCollections.push(checkbox.value);
                    }
                });
                
                // Validate inputs
                if (!queryInput.value.trim()) {
                    alert('Please enter a query');
                    return;
                }
                
                if (selectedCollections.length === 0) {
                    alert('Please select at least one collection');
                    return;
                }
                
                // Show loading
                loading.style.display = 'block';
                responseContainer.style.display = 'none';
                submitButton.disabled = true;
                
                // Prepare request data
                const requestData = {
                    query: queryInput.value.trim(),
                    collections: selectedCollections,
                    chunk_limit: parseInt(chunkLimit.value),
                    temperature: parseFloat(temperature.value),
                    similarity_threshold: parseFloat(similarityThreshold.value)
                };
                
                try {
                    // Send request
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(requestData)
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // Display response
                        geminiResponse.innerHTML = data.response.replace(/\\n/g, '<br>');
                        
                        // Display token information
                        if (data.token_info) {
                            document.getElementById('input-tokens').textContent = data.token_info.input_tokens || 0;
                            document.getElementById('output-tokens').textContent = data.token_info.output_tokens || 0;
                            document.getElementById('total-tokens').textContent = data.token_info.total_tokens || 0;
                        }
                        
                        // Display chunks
                        chunksContainer.innerHTML = '';
                        data.chunks.forEach((chunk, index) => {
                            const chunkDiv = document.createElement('div');
                            chunkDiv.className = 'result-card';
                            
                            let metadataHtml = '';
                            if (chunk.metadata) {
                                const metadataItems = [];
                                for (const [key, value] of Object.entries(chunk.metadata)) {
                                    metadataItems.push(`<span>${key}: ${value}</span>`);
                                }
                                metadataHtml = `<div class="metadata">${metadataItems.join(' | ')}</div>`;
                            }
                            
                            chunkDiv.innerHTML = `
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <span class="collection-tag">${chunk.collection}</span>
                                    <small>Score: ${chunk.score.toFixed(4)}</small>
                                </div>
                                <div>${chunk.text}</div>
                                ${metadataHtml}
                            `;
                            
                            chunksContainer.appendChild(chunkDiv);
                        });
                        
                        responseContainer.style.display = 'block';
                    } else {
                        alert('Error: ' + data.error);
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('An error occurred while processing your request');
                } finally {
                    loading.style.display = 'none';
                    submitButton.disabled = false;
                }
            });
        });
    </script>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    ''')

if __name__ == '__main__':
    # Check if the GOOGLE_API_KEY is set
    if not GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY is not set. Please add it to your .env file.")
    
    # Check if Qdrant client is properly configured
    try:
        collections = get_available_collections()
        print(f"Available collections: {collections}")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
    
    app.run(debug=True)

#To run:
#/Users/mason/opt/anaconda3/envs/psai/bin/python /Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/code/app.py

#Make it sound more like PS - get rid of "as I said"