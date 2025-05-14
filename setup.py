from setuptools import setup, find_packages

setup(
    name="liverag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "boto3==1.35.88",
        "opensearch-py==2.8.0",
        "pinecone==5.4.2",
        "torch==2.5.1",
        "transformers==4.45.2",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.12",
    description="Hybrid Retrieval-Augmented Generation (RAG) pipeline",
)
