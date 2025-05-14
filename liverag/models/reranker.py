from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = None):
        """Initialize the reranker with a specified model.
        
        Args:
            model_name: Name of the model to use for reranking
            device: Device to run the model on (cuda, cpu, or mps)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def rerank(self, query: str, docs: list[dict], top_k: int = 10) -> list[dict]:
        """Rerank documents based on their relevance to the query.
        
        Args:
            query: The search query
            docs: List of documents to rerank, each containing 'doc_id' and 'content'
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with scores
        """
        pairs = [f"{query} [SEP] {doc['content']}" for doc in docs]
        encodings = self.tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**encodings).logits.squeeze(-1).cpu().tolist()

        # Attach scores and sort
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = score
        reranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k] 