# LiveRAG 2025

A hybrid RAG (Retrieval-Augmented Generation) pipeline implementation for LiveRAG 2025 Challenge that combines sparse and dense retrieval methods with reranking for improved question answering.

## Features

- Hybrid search combining BM25 (sparse) and dense vector search
- Reciprocal Rank Fusion (RRF) for merging search results
- Neural reranking using BGE-reranker
- AWS Bedrock integration for LLM inference
- Support for both Pinecone and OpenSearch indices


## Project Structure

```
liverag/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── .gitignore
├── liverag/
│   ├── __init__.py
│   │   ├── indices/
│   │   │   ├── __init__.py
│   │   │   ├── pinecone_client.py
│   │   │   └── opensearch_client.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── reranker.py
│   │   │   └── embeddings.py
│   │   └── pipeline/
│   │       ├── __init__.py
│   │       └── rag_pipeline.py
│   ├── examples/
│   │   ├── README.md
│   │   ├── example_usage.py
│   │   ├── input/
│   │   └── output/
│   └── test-day/
│       └── data
|       ├── __init__.py
|       └── test_pipeline.py
```

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/liverag.git](https://github.com/mallpriyanshu/LiveRAG-2025.git)
cd liverag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## AWS Setup

1. Get your AWS access key and secret key from AWS console:
   - Log in to the AWS Management Console
   - Click on your name at the top-right corner and then "Security Credentials"
   - Click on "Access keys" and create a new access key for CLI
   - Download and save your access key and secret key

2. Install the AWS CLI tool:
   - Follow instructions at: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

3. Configure AWS CLI:
```bash
aws configure --profile sigir-participant
# Use the following settings:
# AWS Access Key ID: [your access key]
# AWS Secret Access Key: [your secret key]
# Default region name: us-east-1
```

4. Test your setup:
```bash
# Should display your AWS account ID
aws sts get-caller-identity --profile sigir-participant

# Verify access to configuration service
aws ssm get-parameter --name /pinecone/ro_token --profile sigir-participant
```

## Usage

Basic usage example:

```python
from liverag.pipeline.rag_pipeline import rag_pipeline

# Run the pipeline
result = rag_pipeline(
    query="What exactly gives particles their mass according to physics?",
    query_id="test_001"
)

# Access results
print(f"Query: {result['query']}")
print(f"Answer: {result['answer']}")
print(f"Execution time: {result['execution_time']:.2f} seconds")
```

For more detailed examples, check the `examples/` directory which contains:
- `example_usage.py`: example of using the RAG pipeline
- `input/`: Directory for input files
- `output/`: Directory for generated outputs
- `README.md`: documentation of the examples

## Reproducing Results

This section provides detailed instructions for reproducing the results of our RAG pipeline on the test queries for LiveRAG 2025 challenge.

### Prerequisites

1. Python 3.12 or higher
2. AWS account with Bedrock access
3. Access to the provided Pinecone and OpenSearch indices
4. Test queries file (`test-set.jsonl`)
5. AWS Bedrock endpoint configuration

### AWS Bedrock Setup

1. **Endpoint Configuration**
   - The pipeline requires a specific AWS Bedrock endpoint for the Falcon-3-10B-Instruct model
   - Update the endpoint ARN in `liverag/pipeline/rag_pipeline.py`:
   ```python
   # AWS Bedrock configuration
   ENDPOINT_ARN = 'your-endpoint-arn'  # Replace with your Falcon3-10B-Instruct endpoint ARN
   ```
   - The endpoint should be in the `us-east-1` region
   - Ensure you have the necessary permissions to access the endpoint
   - The endpoint must be running the Falcon3-10B-Instruct model 

2. **Model Requirements**
   - The pipeline is specifically designed to work with the Falcon3-10B-Instruct model
   - The model should be deployed as a Bedrock endpoint
   - Minimum requirements for the endpoint:
     - Model: Falcon3-10B-Instruct
     - Region: us-east-1
     - Instance type: ml.g5.2xlarge or higher recommended
     

3. **AWS Configuration to access Pinecone and OpenSearch indexes**
```bash
# Configure AWS credentials
aws configure --profile sigir-participant
# Enter the following when prompted:
# AWS Access Key ID: [your access key]
# AWS Secret Access Key: [your secret key]
# Default region name: us-east-1
# Default output format: json
```

4. **Verify Access**
```bash
# Test AWS configuration
aws sts get-caller-identity --profile sigir-participant
aws ssm get-parameter --name /pinecone/ro_token --profile sigir-participant
```

4. **Run the Pipeline**

The test pipeline is located in the `test-day` directory and provides processing all test-set queries:

```bash
cd test-day
python test_pipeline.py
```

The script will:
- Load queries from `data/test-set.jsonl`
- Process each query through the RAG pipeline
- Save results to `data/test-set_output.jsonl`

### Expected Output Format

The output file (`test-set_output.jsonl`) will contain one JSON object per line with the following structure:
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

### Performance Metrics

- Average query processing time: ~15-20 seconds per query
- Total processing time for 100 queries: ~25-30 minutes


### Troubleshooting

1. **AWS Authentication Issues**
   - Verify AWS credentials are correctly configured
   - Ensure access to required AWS services (Bedrock, SSM)
   - Verify the Bedrock endpoint ARN is correctly set in `rag_pipeline.py`
   - Check if the endpoint is in the correct region (us-east-1) if not make changes accordingly in the script.

2. **Bedrock Endpoint Issues**
   - Ensure the Falcon3-10B-Instruct endpoint is active and running
   - Verify you have the necessary IAM permissions
   - Verify the endpoint is running the correct model version
   - Check endpoint metrics for any performance issues


### Contact

For any issues with reproduction or questions about the implementation, please contact:
- Email: mall.priyanshu7@gmail.com
- GitHub Issues: https://github.com/yourusername/liverag/issues

## Configuration

The pipeline uses several configuration parameters that can be customized:

- Pinecone index name and namespace
- OpenSearch index name
- AWS region and profile
- Model parameters for reranking and embeddings
- Inference parameters for the LLM

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

