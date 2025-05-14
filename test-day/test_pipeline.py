import json
from pathlib import Path
from tqdm.auto import tqdm
from liverag.pipeline.rag_pipeline import rag_pipeline

def batch_processing_example(input_file: str = "data/test-set.jsonl", 
                             output_file: str = "data/test-set_output.jsonl"):
    """Batch processing multiple queries for evaluation."""
    
    # Define paths
    test_data_path = Path(input_file)
    output_path = Path(output_file)
    
    # Load test queries
    with test_data_path.open("r", encoding="utf-8") as f:
        test_queries = [json.loads(line) for line in f]
    
    test_queries = test_queries[:5]

    
    # Process queries
    results = []
    for item in tqdm(test_queries, desc="Processing test queries"):
        query = item["question"]
        query_id = item["id"]
        
        result = rag_pipeline(query, query_id)
        
        # Convert to evaluation format
        passages = [
            {
                "passage": doc["text"],
                "doc_IDs": doc.get("id") if isinstance(doc.get("docid"), list) else [doc.get("docid")]
            }
            for doc in result.get("top_docs", [])
        ]
        
        json_entry = {
            "id": query_id,
            "question": query,
            "passages": passages,
            "final_prompt": result.get("final_prompt", ""),
            "answer": result.get("answer", "")
        }
        
        results.append(json_entry)
    
    # Save output
    with output_path.open("w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f)
            f.write("\n")

if __name__ == "__main__":
    print("Running batch queries ...")
    print("-" * 50)
    batch_processing_example()
    print("Output saved to data/test-set_output.jsonl")
