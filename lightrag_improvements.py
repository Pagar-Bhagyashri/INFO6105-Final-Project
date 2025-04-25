import re
import numpy as np
from scipy.spatial.distance import cosine

def apply_improvements(rag):
    """
    Apply improvements to a LightRAG instance
    """
    # Store the original RAG methods
    original_insert = rag.insert
    original_query = rag.query
    
    # 1. Adaptive chunking improvement
    def improved_insert(text, doc_id=None, **kwargs):
        # Apply adaptive chunking
        chunks = adaptive_chunking(text)
        
        # Call the original insert method for each chunk
        doc_ids = []
        for chunk in chunks:
            chunk_doc_id = original_insert(chunk, **kwargs)
            doc_ids.append(chunk_doc_id)
        
        return doc_ids[0] if doc_ids else None
    
    # 2. Enhanced query processing
    def improved_query(query, param=None, **kwargs):
        # Apply query enhancement
        enhanced_query = enhance_query(query)
        
        # Apply dynamic retrieval weighting
        if param:
            param = dynamic_retrieval_weighting(query, param)
        
        # Call original query with improved parameters
        return original_query(enhanced_query, param, **kwargs)
    
    # Replace the methods
    rag.insert = improved_insert
    rag.query = improved_query
    
    return rag

def adaptive_chunking(text, min_chunk_size=150, max_chunk_size=500, overlap_size=50):
    """
    Improved chunking strategy that respects natural language boundaries
    """
    # Split by paragraphs first
    paragraphs = re.split(r'\n\n+', text)
    
    # Initialize chunks
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph exceeds max size and we already have content
        if len(current_chunk) + len(paragraph) > max_chunk_size and len(current_chunk) >= min_chunk_size:
            # Add current chunk to results
            chunks.append(current_chunk)
            
            # Start new chunk with overlap
            sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
            overlap_text = " ".join(sentences[-2:]) if len(sentences) > 2 else ""
            current_chunk = overlap_text + " " + paragraph
        else:
            # Add paragraph to current chunk
            current_chunk += " " + paragraph if current_chunk else paragraph
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def enhance_query(query):
    """
    Enhance query with contextual information
    """
    # Extract key terms
    key_terms = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', query)
    
    # Add synonyms for key terms (simplified version)
    enhanced_terms = set(key_terms)
    for term in key_terms:
        if term.lower() == "find":
            enhanced_terms.add("locate")
            enhanced_terms.add("discover")
        elif term.lower() == "explain":
            enhanced_terms.add("describe")
            enhanced_terms.add("elaborate")
    
    # Build enhanced query by adding key terms
    enhanced_query = query
    if len(enhanced_terms) > len(key_terms):
        enhanced_query += " (Related terms: " + ", ".join(enhanced_terms - set(key_terms)) + ")"
    
    return enhanced_query

def dynamic_retrieval_weighting(query, param):
    """
    Dynamically adjust retrieval parameters based on query characteristics
    """
    # Check if query is factual (who, what, when, where, why, how)
    is_factual = bool(re.search(r'\b(who|what|when|where|why|how)\b', query.lower()))
    
    # Check if query has named entities
    has_entities = bool(re.findall(r'\b[A-Z][a-z]+\b', query))
    
    # Adjust top_k based on query characteristics
    if is_factual and has_entities:
        param.top_k = max(param.top_k, 5)  # Increase for specific factual queries
    elif is_factual:
        param.top_k = max(param.top_k, 4)  # Slight increase for general factual queries
    
    # Adjust mode based on query
    if is_factual and not has_entities:
        param.mode = "semantic"  # Favor semantic for conceptual queries
    elif has_entities:
        param.mode = "hybrid"  # Use hybrid for entity queries
    
    return param