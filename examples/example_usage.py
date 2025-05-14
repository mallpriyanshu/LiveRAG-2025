import json
from pathlib import Path
from tqdm.auto import tqdm
from liverag.pipeline.rag_pipeline import rag_pipeline

def single_query_example():
    """Example of processing a single query."""
    # Example query
    example_query = "What exactly gives particles their mass according to physics?"
    
    # Run pipeline
    result = rag_pipeline(example_query)
    
    # Print results
    print(f"Query: {result['query']}")
    print(f"\nExecution time: {result['execution_time']:.2f} seconds")
    
    print("\nTop documents:")
    for i, doc in enumerate(result['top_docs'], 1):
        print(f"\n{i}. Score: {doc['score']:.4f}")
        print(f"ID: {doc['id']}")
        print(f"Text: {doc['text'][:200]}...")
        print(f'Doc Length: {len(doc["text"])}')
    
    print("\nRelevant chunks:")
    print(result['relevant_chunks'])
    
    print("\nFinal answer:")
    print(result['answer'])

def batch_processing_example(input_file: str = "input/questions-0-99.jsonl", 
                           output_file: str = "output/answers_output.jsonl"):
    """Example of batch processing multiple queries."""
    # Define paths
    test_data_path = Path(input_file)
    output_path = Path(output_file)
    
    # Load test data
    test_queries = []
    with test_data_path.open("r", encoding="utf-8") as f:
        for line in f:
            test_queries.append(json.loads(line))
    
    test_queries = test_queries[:5]  #comment this line to run all queries

    # Run pipeline and collect results
    results = []
    for item in tqdm(test_queries, desc="Processing test queries"):
        query = item["question"]
        query_id = item["id"]
        
        result = rag_pipeline(query, query_id)
        
        # Convert top_docs to required passage format
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
    
    # Write output to JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f)
            f.write("\n")

if __name__ == "__main__":
    print("Running single query ...")
    print("-" * 50)
    single_query_example()
    
    print("\n\nRunning batch queries...")
    print("-" * 50)
    batch_processing_example() 
    print("Output saved to output/answers_output.jsonl")