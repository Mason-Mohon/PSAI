from flask import Flask, render_template, request, jsonify, send_file
import os
from dotenv import load_dotenv
from pathlib import Path
import qdrant_client
from qdrant_client.models import Distance
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import json
from typing import TypedDict, List, Dict, Any, Literal
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv("/Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/code/.env")

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

# Initialize clients
qdrant_client_instance = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
genai_client_instance = genai.Client(api_key=GOOGLE_API_KEY)

# --- LangGraph State Definition ---
class GraphState(TypedDict):
    original_query: str
    refined_query: str | None
    selected_collections: List[str]
    
    initial_chunk_limit: int
    current_chunk_limit: int
    max_chunk_limit: int
    
    similarity_threshold: float
    temperature: float
    
    search_results: List[Dict[str, Any]] | None
    formatted_context_for_generation: str | None
    
    generated_response_text: str | None
    token_info: Dict[str, int] | None
    
    critique_json: Dict[str, str] | None
    
    iteration_count: int
    max_iterations: int
    
    final_json_response: Dict[str, Any] | None
    error_message: str | None


# --- Existing Functions (adapted slightly if needed for graph) ---
def get_available_collections_internal():
    collections = [c.name for c in qdrant_client_instance.get_collections().collections]
    return collections

def refine_query_for_semantic_search_internal(query_text: str) -> str:
    prompt = f"""Rewrite the following user query to be optimized for semantic search against a knowledge base primarily focused on Phyllis Schlafly's life, work, and conservative viewpoints.
Extract the key entities, topics, and the core intent. Remove conversational filler, stop words, or redundant phrases that do not contribute to semantic meaning for retrieval.
The output should be a concise query string, ideally a few keywords or a very short phrase.

User Query: "{query_text}"
Optimized Search Query:"""
    try:
        response = genai_client_instance.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=60,
                stop_sequences=["\n"]
            )
        )
        
        # Check if response and response.text are valid
        if not response or not response.text:
            print("Warning: Empty response from Gemini API during query refinement")
            return query_text
            
        refined_query = response.text.strip()
        if refined_query.startswith("Optimized Search Query:"):
            refined_query = refined_query.replace("Optimized Search Query:", "").strip()
        refined_query = refined_query.strip('\'\"')
        print(f"Original query for refinement: '{query_text}'")
        print(f"Refined query for search: '{refined_query}'")
        return refined_query if refined_query else query_text
    except Exception as e:
        print(f"Error refining query: {e}")
        return query_text

def semantic_search_internal(query_text, collections, limit=5, similarity_threshold=0.0):
    query_vector = embedding_model.encode(query_text).tolist()
    all_results = []
    for collection_name in collections:
        try:
            search_results = qdrant_client_instance.search(
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
                    "ref_id": f"REF_SEM_{idx+1}"
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

def generate_gemini_response_internal(query, context_chunks, temperature=0.7):
    try:
        formatted_chunks_for_prompt = []
        for idx, chunk in enumerate(context_chunks):
            metadata = chunk.get('metadata', {})
            author = metadata.get('author', 'Unknown')
            book_title = metadata.get('book_title', '')
            publication_year = metadata.get('publication_year', '')
            doc_type = metadata.get('doc_type', '')
            source_file = metadata.get('source_file', '')

            source_info = f"Collection: {chunk['collection']}"
            if book_title: source_info += f", Book: {book_title}"
            if publication_year: source_info += f", Year: {publication_year}"
            if author: source_info += f", Author: {author}"
            if doc_type: source_info += f", Type: {doc_type}"

            formatted_chunks_for_prompt.append(f"[{chunk['ref_id']}] Source [{source_info}]: {chunk['text']}")
        
        formatted_context_string = "\\n\\n".join(formatted_chunks_for_prompt)

        system_instruction = (
            "You are playing the role of Phyllis Schlafly answering questions based solely on the provided context. "
            "If the context lacks sufficient detail to answer, say so. Use endnotes for citations but omit source filenames. "
            "If the source is your own writing, speak in your voice as if it is your own. Present a confident, conservative tone."
        )
        prompt = (
            f"Context:\\n{formatted_context_string}\\n\\n"
            f"Question: {query}\\n\\n"
            "Answer the question strictly based on the above context. "
            "When referencing any chunk, indicate its reference ID inline (e.g., [REF_SEM_1], [REF_SEM_2]) and include numbered endnotes at the end of your response. "
            "Each endnote should follow this format: [n] Title of piece, publication (e.g., Phyllis Schlafly Report or book title), date, author. "
            "If the information is not visible in the chunk itself, use the information within the metadata to generate the citation."
            "Do not include source filenames. If the author is Phyllis Schlafly, treat it as your own words and omit the author from the endnote."
        )
        
        # Estimate input tokens as fallback (1 token ≈ 4 characters)
        input_token_count = len(prompt) // 4

        response = genai_client_instance.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
                max_output_tokens=1024,
            ),
        )
        
        # Check if response and response.text are valid
        if not response or not response.text:
            print("Warning: Empty or None response from Gemini API")
            return {
                "text": "I apologize, but I was unable to generate a response at this time. Please try again.",
                "token_info": {"input_tokens": input_token_count, "output_tokens": 0, "total_tokens": input_token_count},
                "formatted_context_for_generation": formatted_context_string
            }
        
        # Estimate output tokens as fallback (1 token ≈ 4 characters)
        output_token_count = len(response.text) // 4
        
        return {
            "text": response.text,
            "token_info": {
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count
            },
            "formatted_context_for_generation": formatted_context_string
        }
    except Exception as e:
        print(f"Error in generate_gemini_response_internal: {e}")
        return {"text": f"Error generating response: {str(e)}", "token_info": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "formatted_context_for_generation": ""}

# --- LangGraph Nodes ---
def initialize_state_node(state: GraphState) -> Dict[str, Any]:
    print("--- Running: Initialize State Node ---")
    # Extract values directly from the state (which contains the initial input)
    initial_query = state.get('original_query', '')
    error_msg = None
    if not initial_query:
        print("!!! initialize_state_node: Input query is empty. Setting error_message.")
        error_msg = "Query is required in the input."

    return {
        "original_query": initial_query,
        "selected_collections": state.get('selected_collections', []),
        "initial_chunk_limit": int(state.get('initial_chunk_limit', 5)),
        "current_chunk_limit": int(state.get('current_chunk_limit', 5)),
        "max_chunk_limit": 15,
        "similarity_threshold": float(state.get('similarity_threshold', 0.0)),
        "temperature": float(state.get('temperature', 0.7)),
        "iteration_count": 1,
        "max_iterations": 3,
        "refined_query": None,
        "search_results": None,
        "formatted_context_for_generation": None,
        "generated_response_text": None,
        "token_info": None,
        "critique_json": None,
        "final_json_response": None,
        "error_message": error_msg
    }

def refine_query_node(state: GraphState) -> Dict[str, Any]:
    print(f"--- Running: Refine Query Node (Iteration: {state['iteration_count']}) ---")
    if not state['original_query']:
        print("!!! refine_query_node: Setting error_message - Query is required")
        return {"error_message": "Query is required"}
    refined = refine_query_for_semantic_search_internal(state['original_query'])
    return {"refined_query": refined}

def semantic_search_node(state: GraphState) -> Dict[str, Any]:
    print(f"--- Running: Semantic Search Node (Iteration: {state['iteration_count']}, Chunks: {state['current_chunk_limit']}) ---")
    if not state['refined_query'] or not state['selected_collections']:
        print("!!! semantic_search_node: Setting error_message - Refined query or collections missing")
        return {"error_message": "Refined query or collections missing for search"}
    
    results = semantic_search_internal(
        query_text=state['refined_query'],
        collections=state['selected_collections'],
        limit=state['current_chunk_limit'],
        similarity_threshold=state['similarity_threshold']
    )
    return {"search_results": results}

def generate_response_node(state: GraphState) -> Dict[str, Any]:
    print(f"--- Running: Generate Response Node (Iteration: {state['iteration_count']}) ---")
    if state['search_results'] is None:
        print("--- generate_response_node: No search results found.")
        return {"generated_response_text": "No search results to generate a response from.", 
                "token_info": {"input_tokens":0, "output_tokens":0, "total_tokens":0},
                "formatted_context_for_generation": ""}

    if not state['original_query']:
         print("!!! generate_response_node: Setting error_message - Original query missing")
         return {"error_message": "Original query missing for generation"}

    response_data = generate_gemini_response_internal(
        query=state['original_query'],
        context_chunks=state['search_results'],
        temperature=state['temperature']
    )
    return {
        "generated_response_text": response_data["text"],
        "token_info": response_data["token_info"],
        "formatted_context_for_generation": response_data["formatted_context_for_generation"]
    }

def critique_response_node(state: GraphState) -> Dict[str, Any]:
    print(f"--- Running: Critique Response Node (Iteration: {state['iteration_count']}) ---")
    if not state['generated_response_text'] or not state['original_query']:
        print("--- critique_response_node: Missing generated_response_text or original_query for critique.")
        return {"critique_json": {"answer_quality": "ERROR_IN_CRITIQUE", "reasoning": "Missing generated response or query for critique."}}

    context_str = state.get('formatted_context_for_generation', "Context not available.")
    if not state['search_results']:
        context_str = "No context was retrieved or provided for generation."

    critique_prompt = f"""You are an expert evaluator. Your task is to assess a generated answer based on a user's query and the context retrieved to formulate that answer.
The system can retrieve up to {state['max_chunk_limit']} context chunks in total. It is currently on iteration {state['iteration_count']} of {state['max_iterations']} and has retrieved {state['current_chunk_limit']} chunks.

User Query:
{state['original_query']}

Retrieved Context Used for Answer:
{context_str}

Generated Answer:
{state['generated_response_text']}

Based on this, provide your evaluation in JSON format with the following keys:
- "answer_quality": A string, must be one of ["GOOD", "ACCEPTABLE_NEEDS_MORE_CONTEXT", "POOR_NEEDS_MORE_CONTEXT", "ACCEPTABLE_NO_MORE_CONTEXT_NEEDED", "POOR_NO_MORE_CONTEXT_NEEDED"].
  - "GOOD": The answer is comprehensive and well-supported. No further action needed.
  - "ACCEPTABLE_NEEDS_MORE_CONTEXT": The answer is okay but could be substantially improved with more supporting details from the knowledge base, and more chunks might be available.
  - "POOR_NEEDS_MORE_CONTEXT": The answer is weak/incomplete, and more context is likely required and might be available.
  - "ACCEPTABLE_NO_MORE_CONTEXT_NEEDED": The answer is okay. More context is unlikely to help, is not available, or we've hit chunk/iteration limits.
  - "POOR_NO_MORE_CONTEXT_NEEDED": The answer is weak. More context is unlikely to help, is not available, or we've hit chunk/iteration limits.
- "reasoning": A brief explanation for your assessment.

JSON Output:
"""
    try:
        response = genai_client_instance.models.generate_content(
            model=GEMINI_MODEL,
            contents=critique_prompt,
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=200)
        )
        
        # Check if response and response.text are valid
        if not response or not response.text:
            print("Warning: Empty response from Gemini API during critique")
            return {"critique_json": {"answer_quality": "ERROR_IN_CRITIQUE", "reasoning": "Empty response from critique API"}}
            
        critique_text = response.text.strip()
        if critique_text.startswith("```json"):
            critique_text = critique_text[len("```json"):]
        if critique_text.endswith("```"):
            critique_text = critique_text[:-len("```")]
        critique_text = critique_text.strip()
        
        parsed_critique = json.loads(critique_text)
        print(f"Critique received and parsed: {parsed_critique}")
        return {"critique_json": parsed_critique}
    except Exception as e:
        print(f"!!! critique_response_node: Error during critique API call or JSON parsing: {e}. Raw critique response: '{response.text if 'response' in locals() else 'N/A'}'")
        return {"critique_json": {"answer_quality": "ERROR_IN_CRITIQUE", "reasoning": str(e)}}

def prepare_final_output_node(state: GraphState | None) -> Dict[str, Any]:
    print("--- Running: Prepare Final Output Node ---")
    if state is None:
        print("!!! prepare_final_output_node: Critical Error - Received None as state!")
        return {
            "final_json_response": {
                "error": "Critical graph error: State was lost before final output processing.",
                "original_query": "Unknown (state was None)",
                "response": "Critical graph error: State was None during final output.",
                "iterations_done": 0,
                "final_chunk_limit_used": 0,
                "critique_assessment": "N/A (state was None)",
                "critique_reasoning": "N/A (state was None)"
            }
        }

    original_query = state.get('original_query', 'Unknown - original query missing from state')
    
    final_response = {
        "original_query": original_query,
        "refined_query_for_search": state.get('refined_query', original_query),
        "query_used_for_search": state.get('refined_query', original_query),
        "chunks": state.get('search_results', []),
        "response": "Error: Processing did not complete successfully.", # Default error response
        "token_info": state.get('token_info', {}),
        "iterations_done": state.get('iteration_count', 0), # Default to 0 if not properly set
        "final_chunk_limit_used": state.get('current_chunk_limit', state.get('initial_chunk_limit', 0)),
        "critique_assessment": "N/A", # Default
        "critique_reasoning": "N/A"  # Default
    }

    error_msg_from_state = state.get("error_message")
    if error_msg_from_state:
        final_response["error"] = error_msg_from_state
        final_response["response"] = error_msg_from_state
    else:
        # Only try to get full details if no primary error_message was found
        generated_response_text = state.get('generated_response_text')
        if generated_response_text:
            final_response["response"] = generated_response_text

        critique_json_val = state.get('critique_json')
        if critique_json_val: 
            final_response["critique_assessment"] = critique_json_val.get('answer_quality', 'N/A - critique data malformed')
            final_response["critique_reasoning"] = critique_json_val.get('reasoning', 'N/A - critique data malformed')
        elif not final_response.get("error"): # If no primary error and no critique, note it wasn't critiqued
            final_response["critique_assessment"] = "Not Critiqued"

    return {"final_json_response": final_response}


# --- Conditional Edge Logic ---
def should_retry_search_edge(state: GraphState | None) -> str:
    print(f"--- Condition: Router --- ") # Simpler log
    if state is None:
        print("!!! Router: Critical Error - Received None as state! Routing to prepare_final_output.")
        # Cannot set error_message if state is None. Just route.
        return "prepare_final_output"
    
    iteration_count = state.get("iteration_count", 0)
    print(f"Router: Iteration {iteration_count}")

    current_error_message = state.get("error_message")
    print(f"Router: Current error_message in state: {current_error_message}")
    if current_error_message:
        print(f"Router: Error detected ('{current_error_message}'), going to prepare_final_output.")
        return "prepare_final_output"

    # Determine which step we just completed or should go to next
    if state.get("critique_json") is not None:
        print("Router: Path based on critique_json.")
        critique = state['critique_json'] # critique_json is confirmed not None
        # Ensure critique is a dictionary before .get (critique_response_node should ensure this)
        if not isinstance(critique, dict):
            print("!!! Router: critique_json is not a dict. Forcing error for final output.")
            state['error_message'] = "Internal Error: Critique data malformed."
            return "prepare_final_output"

        quality = critique.get("answer_quality", "ERROR_IN_CRITIQUE")
        print(f"Router: Critique quality: {quality}")

        if quality in ["GOOD", "ACCEPTABLE_NO_MORE_CONTEXT_NEEDED", "POOR_NO_MORE_CONTEXT_NEEDED", "ERROR_IN_CRITIQUE"]:
            print("Router: Quality sufficient or no more retries. Preparing final output.")
            return "prepare_final_output"
        
        if iteration_count >= state.get('max_iterations', 3):
            print("Router: Max iterations reached. Preparing final output.")
            return "prepare_final_output"

        if quality in ["ACCEPTABLE_NEEDS_MORE_CONTEXT", "POOR_NEEDS_MORE_CONTEXT"]:
            if state.get('current_chunk_limit', 0) >= state.get('max_chunk_limit', 15):
                print("Router: Max chunk limit reached. Preparing final output.")
                return "prepare_final_output"
            else:
                print("Router: Retrying. Going to update_state_for_retry.")
                return "update_state_for_retry"
        
        print("Router: Unexpected critique assessment. Preparing final output.")
        return "prepare_final_output"
    
    elif state.get("generated_response_text") is not None:
        print("Router: Path based on generated_response_text. Proceeding to critique.")
        return "critique_response"
    
    elif state.get("search_results") is not None:
        print("Router: Path based on search_results. Proceeding to generate_response.")
        return "generate_response"
    
    elif state.get("refined_query") is not None:
        print("Router: Path based on refined_query. Proceeding to semantic_search.")
        return "semantic_search"
    
    elif state.get("original_query") is not None: # Initial state after successful init and no error
        print("Router: Path based on original_query. Proceeding to refine_query.")
        return "refine_query"
    
    else: 
        print("!!! Router: Critical state error - original_query is missing and no other path taken. Setting error.")
        state['error_message'] = "Critical error: Graph state unclear, original_query missing."
        return "prepare_final_output"

def update_state_for_retry_node(state: GraphState) -> Dict[str, Any]:
    print(f"--- Running: Update State for Retry Node (Old Iteration: {state['iteration_count']}) ---")
    new_chunk_limit = min(state['current_chunk_limit'] + 5, state['max_chunk_limit'])
    return {
        "current_chunk_limit": new_chunk_limit,
        "iteration_count": state['iteration_count'] + 1,
        "search_results": None,
        "generated_response_text": None,
        "token_info": None,
        "critique_json": None
    }

# --- Build the Graph ---
workflow = StateGraph(GraphState)

# Add ACTUAL processing nodes that update state
workflow.add_node("initialize_state", lambda state: initialize_state_node(state))
workflow.add_node("refine_query", refine_query_node)
workflow.add_node("semantic_search", semantic_search_node)
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("critique_response", critique_response_node)
workflow.add_node("update_state_for_retry", update_state_for_retry_node)
workflow.add_node("prepare_final_output", prepare_final_output_node)

# Set entry point
workflow.set_entry_point("initialize_state")

# Define the path map for conditional edges
PATH_MAP = {
    "refine_query": "refine_query",
    "semantic_search": "semantic_search",
    "generate_response": "generate_response",
    "critique_response": "critique_response",
    "update_state_for_retry": "update_state_for_retry",
    "prepare_final_output": "prepare_final_output",
    END: END
}

# After each processing node, decide where to go next using should_retry_search_edge
workflow.add_conditional_edges("initialize_state", should_retry_search_edge, PATH_MAP)
workflow.add_conditional_edges("refine_query", should_retry_search_edge, PATH_MAP)
workflow.add_conditional_edges("semantic_search", should_retry_search_edge, PATH_MAP)
workflow.add_conditional_edges("generate_response", should_retry_search_edge, PATH_MAP)
workflow.add_conditional_edges("critique_response", should_retry_search_edge, PATH_MAP)

# Specific non-conditional edges
workflow.add_edge("update_state_for_retry", "semantic_search") # Loop back to search
workflow.add_edge("prepare_final_output", END) # Final step

# Compile the graph
app_graph = workflow.compile()


# --- Flask Routes ---
@app.route('/')
def index():
    collections = get_available_collections_internal()
    return render_template('index.html', collections=collections)

@app.route('/api/query', methods=['POST'])
def query_api_route():
    data = request.json
    
    initial_graph_input = {
        "original_query": data.get('query', ''),
        "selected_collections": data.get('collections', []),
        "initial_chunk_limit": int(data.get('chunk_limit', 5)),
        "similarity_threshold": float(data.get('similarity_threshold', 0.0)),
        "temperature": float(data.get('temperature', 0.7)),
        "current_chunk_limit": int(data.get('chunk_limit', 5)),
        "max_chunk_limit": 15,
        "iteration_count": 1,
        "max_iterations": 3,
    }
    
    if not initial_graph_input["original_query"]:
        return jsonify({"error": "Query is required"}), 400
    if not initial_graph_input["selected_collections"]:
        return jsonify({"error": "At least one collection must be selected"}), 400

    final_state = app_graph.invoke(initial_graph_input)
    
    if final_state.get("final_json_response"):
        return jsonify(final_state["final_json_response"])
    else:
        error_msg = final_state.get("error_message", "An unexpected error occurred in the graph processing.")
        return jsonify({
            "error": error_msg,
            "original_query": initial_graph_input["original_query"],
            "response": error_msg
        }), 500

def format_conversation_text(query: str, response: str, chunks: List[Dict]) -> str:
    """Format the conversation as plain text."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"Conversation Export - {timestamp}\n\n"
    text += "Question:\n" + query + "\n\n"
    text += "Answer:\n" + response + "\n\n"
    text += "Reference Chunks:\n"
    
    for i, chunk in enumerate(chunks, 1):
        text += f"\nChunk {i}:\n"
        text += f"Collection: {chunk.get('collection', 'N/A')}\n"
        text += f"Text: {chunk.get('text', 'N/A')}\n"
        text += f"Score: {chunk.get('score', 'N/A')}\n"
        
        metadata = chunk.get('metadata', {})
        if metadata:
            text += "Metadata:\n"
            for key, value in metadata.items():
                text += f"  {key}: {value}\n"
        text += "-" * 80 + "\n"
    
    return text

def create_pdf(query: str, response: str, chunks: List[Dict]) -> bytes:
    """Create a PDF document of the conversation."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12
    )
    normal_style = styles['Normal']
    
    # Content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements = []
    
    # Title
    elements.append(Paragraph(f"Conversation Export - {timestamp}", title_style))
    elements.append(Spacer(1, 12))
    
    # Question
    elements.append(Paragraph("Question:", heading_style))
    elements.append(Paragraph(query, normal_style))
    elements.append(Spacer(1, 12))
    
    # Answer
    elements.append(Paragraph("Answer:", heading_style))
    elements.append(Paragraph(response, normal_style))
    elements.append(Spacer(1, 12))
    
    # Reference Chunks
    elements.append(Paragraph("Reference Chunks:", heading_style))
    
    for i, chunk in enumerate(chunks, 1):
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Chunk {i}:", heading_style))
        
        # Create a table for chunk details
        data = [
            ["Collection:", chunk.get('collection', 'N/A')],
            ["Score:", str(chunk.get('score', 'N/A'))],
            ["Text:", chunk.get('text', 'N/A')]
        ]
        
        metadata = chunk.get('metadata', {})
        for key, value in metadata.items():
            data.append([f"{key}:", str(value)])
            
        table = Table(data, colWidths=[1.5*inch, 5*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        elements.append(table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

@app.route('/api/download/<format>', methods=['POST'])
def download_conversation(format):
    data = request.json
    query = data.get('query', '')
    response = data.get('response', '')
    chunks = data.get('chunks', [])
    
    if format not in ['txt', 'pdf']:
        return jsonify({"error": "Invalid format. Must be 'txt' or 'pdf'"}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == 'txt':
        text_content = format_conversation_text(query, response, chunks)
        buffer = io.BytesIO(text_content.encode('utf-8'))
        filename = f"conversation_{timestamp}.txt"
        mimetype = 'text/plain'
    else:  # pdf
        buffer = io.BytesIO(create_pdf(query, response, chunks))
        filename = f"conversation_{timestamp}.pdf"
        mimetype = 'application/pdf'
    
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    if not GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY is not set. Please add it to your .env file.")

    try:
        collections = get_available_collections_internal()
        print(f"Available collections: {collections}")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")

    app.run(debug=True)

#To run:
#/Users/mason/opt/anaconda3/envs/psai/bin/python /Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/code/app2.py