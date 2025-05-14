# LiveRAG Examples

This directory contains example scripts demonstrating how to use the LiveRAG pipeline.

## Example Usage

The `example_usage.py` script demonstrates two ways to use the LiveRAG pipeline:

1. Single Query Processing
2. Batch Processing of Multiple Queries

### Running the Examples

1. Make sure you have installed the package and set up AWS credentials as described in the main README.

2. Run the example script:
```bash
python example_usage.py
```

### Single Query Example

The single query example demonstrates how to process one query and view the detailed results:

```python
from liverag.pipeline.rag_pipeline import rag_pipeline

# Example query
result = rag_pipeline("What exactly gives particles their mass according to physics?")

# Access results
print(f"Query: {result['query']}")
print(f"Answer: {result['answer']}")
print(f"Execution time: {result['execution_time']:.2f} seconds")
```

### Batch Processing Example

The batch processing example shows how to process multiple queries from a JSONL file and save the results:

```python
from liverag.pipeline.rag_pipeline import rag_pipeline
import json
from pathlib import Path

# Load queries from JSONL file
with Path("questions-0-99.jsonl").open("r") as f:
    queries = [json.loads(line) for line in f]

# Process each query
results = []
for item in queries:
    result = rag_pipeline(item["question"], item["id"])
    results.append(result)

# Save results
with Path("output.jsonl").open("w") as f:
    for result in results:
        json.dump(result, f)
        f.write("\n")
```

## Input/Output Format

### Input JSONL Format
Each line in the input JSONL file should contain:
```json
{
    "id": "query_id",
    "question": "Your question here"
}
```

### Output JSONL Format
Each line in the output JSONL file contains:
```json
{
    "id": "query_id",
    "question": "Original question",
    "passages": [
        {
            "passage": "Retrieved text",
            "doc_IDs": ["document_id"]
        }
    ],
    "final_prompt": "Prompt used for answer generation",
    "answer": "Generated answer"
}
```

## Notes

- Make sure you have the required input files in the correct location
- The script will create output files in the current directory
- Processing time will depend on the number of queries and your system's capabilities
- AWS credentials must be properly configured for the pipeline to work 