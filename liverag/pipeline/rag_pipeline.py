import time
from typing import Dict, Any, List
from collections import defaultdict
from operator import itemgetter

import boto3
from ..indices.pinecone_client import query_pinecone
from ..indices.opensearch_client import query_opensearch
from ..models.reranker import Reranker

# AWS Bedrock configuration
ENDPOINT_ARN = 'your-endpoint-arn'
INFERENCE_CONFIG = {
    "maxTokens": 256,
    "temperature": 0.6,
    "topP": 0.9,
}
ADDITIONAL_MODEL_FIELDS = {
    "parameters": {
        "repetition_penalty": 1.15,
        "top_k": 50,
        "do_sample": True
    }
}

def rrf_score(rank: int, k: int = 60) -> float:
    """Calculate Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)

def rrf_fusion_chunks(dense_chunks: List[Dict], sparse_chunks: List[Dict], k: int = 60) -> List[Dict]:
    """Merge dense and sparse search results using RRF."""
    rrf_scores = defaultdict(float)
    chunk_map = {}

    for rank, chunk in enumerate(dense_chunks):
        chunk_id = chunk['id']
        rrf_scores[chunk_id] += rrf_score(rank, k)
        chunk_map[chunk_id] = {
            'id': chunk['id'],
            'score': chunk['score'],
            'text': chunk['metadata'].get('text', ''),
            'docid': chunk['metadata'].get('doc_id'),
            'source': 'dense'
        }

    for rank, chunk in enumerate(sparse_chunks):
        chunk_id = chunk['_id']
        rrf_scores[chunk_id] += rrf_score(rank, k)
        chunk_map[chunk_id] = {
            'id': chunk['_id'],
            'score': chunk['_score'],
            'text': chunk['_source'].get('text', ''),
            'docid': chunk['_source'].get('doc_id'),
            'source': 'sparse'
        }

    # Sort by RRF score
    sorted_chunks = sorted(rrf_scores.items(), key=itemgetter(1), reverse=True)
    return [chunk_map[chunk_id] for chunk_id, _ in sorted_chunks[:k]]

def merge_search_results_with_reranking(query: str, top_k: int = 5, rrf_k: int = 100) -> List[Dict[str, Any]]:
    """Perform hybrid search with reranking."""
    # Step 1: Query dense and sparse retrievers
    dense = query_pinecone(query, top_k=rrf_k)
    sparse = query_opensearch(query, top_k=rrf_k)

    # Step 2: RRF fusion at chunk level
    fused_chunks = rrf_fusion_chunks(dense["matches"], sparse["hits"]["hits"], k=rrf_k)

    # Step 3: Rerank using full chunk text
    reranker_input = [{"doc_id": chunk["id"], "content": chunk["text"]} for chunk in fused_chunks]
    reranker = Reranker()
    reranked = reranker.rerank(query, reranker_input, top_k=top_k)

    # Step 4: Match reranked scores with original metadata
    chunk_meta_map = {chunk["id"]: chunk for chunk in fused_chunks}

    final_results = []
    for item in reranked:
        chunk_id = item["doc_id"]
        if chunk_id in chunk_meta_map:
            entry = chunk_meta_map[chunk_id].copy()
            entry["score"] = item.get("score", entry["score"])
            final_results.append(entry)

    return final_results

def get_falcon_response_for_extraction(prompt: str) -> str:
    """Get response from Falcon model for chunk extraction."""
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
    response = bedrock_runtime.converse(
        modelId=ENDPOINT_ARN,
        messages=[
            {
                'role': 'user',
                'content': [{'text': prompt}]
            }
        ],
        inferenceConfig={'maxTokens': 1000, 'temperature': 0.6},
        additionalModelRequestFields=ADDITIONAL_MODEL_FIELDS
    )
    return response['output']['message']['content'][0]['text']

def get_falcon_response(prompt: str) -> str:
    """Get response from Falcon model for answer generation."""
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
    response = bedrock_runtime.converse(
        modelId=ENDPOINT_ARN,
        messages=[
            {
                'role': 'user',
                'content': [{'text': prompt}]
            }
        ],
        inferenceConfig=INFERENCE_CONFIG,
        additionalModelRequestFields=ADDITIONAL_MODEL_FIELDS
    )
    return response['output']['message']['content'][0]['text']

def rag_pipeline(query: str, query_id: str = None) -> Dict[str, Any]:
    """Execute the complete RAG pipeline for a given query.
    
    Args:
        query: The question to answer
        query_id: Optional identifier for the query
        
    Returns:
        Dictionary containing:
        - query_id: Identifier for the query
        - query: Original query
        - top_docs: Top retrieved documents
        - relevant_chunks: Extracted relevant chunks
        - final_prompt: Prompt used for answer generation
        - answer: Generated answer
        - execution_time: Total pipeline execution time
    """
    start_time = time.time()
    
    # 1-2. Get sparse search results
    sparse_results = query_opensearch(query, top_k=10)
    
    # 3. Get dense search results
    dense_results = query_pinecone(query, top_k=10)
    
    # 4-5. Merge results and take top n
    top_docs = merge_search_results_with_reranking(query, top_k=5)
    
    # 6. Prompt Falcon for relevant chunks
    chunk_prompt = f"""Given the following query and document chunks, extract all relevant sentences in each new line that help answer the query or are relevant to answering the query.

Query: {query}

Document chunks:"""

    for doc in top_docs:
        text = doc['text'].strip().replace('\n', ' ')  # Clean up whitespace
        chunk_prompt += f"\n---\n{text}"
    
    relevant_chunks = get_falcon_response_for_extraction(chunk_prompt)
    
    # 8. Generate final answer
    answer_prompt = f"""Based on the following relevant information, provide a concise and accurate answer to the query within 300 tokens. Use only the information provided.

Query: {query}

Relevant information:
{relevant_chunks.strip()}"""
    
    final_answer = get_falcon_response(answer_prompt)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # 9. Prepare results
    result = {
        'query_id': query_id or str(int(time.time())),
        'query': query,
        'top_docs': top_docs,
        'relevant_chunks': relevant_chunks,
        'final_prompt': answer_prompt,
        'answer': final_answer,
        'execution_time': execution_time
    }
    
    return result 