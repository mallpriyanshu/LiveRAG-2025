from functools import cache
import boto3
from pinecone import Pinecone
import torch
from transformers import AutoModel, AutoTokenizer

PINECONE_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
PINECONE_NAMESPACE = "default"
AWS_PROFILE_NAME = "sigir-participant"
AWS_REGION_NAME = "us-east-1"

def get_ssm_secret(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
    """Get a secret value from AWS SSM."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ssm = session.client("ssm")
    return ssm.get_parameter(Name=key, WithDecryption=True)["Parameter"]["Value"]

@cache
def has_mps():
    return torch.backends.mps.is_available()

@cache
def has_cuda():
    return torch.cuda.is_available()

@cache
def get_tokenizer(model_name: str = "intfloat/e5-base-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

@cache
def get_model(model_name: str = "intfloat/e5-base-v2"):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if has_mps():
        model = model.to("mps")
    elif has_cuda():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    return model

def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embed_query(query: str,
                query_prefix: str = "query: ",
                model_name: str = "intfloat/e5-base-v2",
                pooling: str = "avg",
                normalize: bool = True) -> list[float]:
    return batch_embed_queries([query], query_prefix, model_name, pooling, normalize)[0]

def batch_embed_queries(queries: list[str], 
                       query_prefix: str = "query: ", 
                       model_name: str = "intfloat/e5-base-v2", 
                       pooling: str = "avg", 
                       normalize: bool = True) -> list[list[float]]:
    with_prefixes = [" ".join([query_prefix, query]) for query in queries]
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)
    with torch.no_grad():
        encoded = tokenizer(with_prefixes, padding=True, return_tensors="pt", truncation="longest_first")
        encoded = encoded.to(model.device)
        model_out = model(**encoded)
        if pooling == "cls":
            embeddings = model_out.last_hidden_state[:, 0]
        else:  # avg
            embeddings = average_pool(model_out.last_hidden_state, encoded["attention_mask"])
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()

@cache
def get_pinecone_index(index_name: str = PINECONE_INDEX_NAME):
    pc = Pinecone(api_key=get_ssm_secret("/pinecone/ro_token"))
    index = pc.Index(name=index_name)
    return index

def query_pinecone(query: str, top_k: int = 1, namespace: str = PINECONE_NAMESPACE) -> dict:
    """Query a Pinecone index and return the results."""
    index = get_pinecone_index()
    results = index.query(
        vector=embed_query(query),
        top_k=top_k,
        include_values=False,
        namespace=namespace,
        include_metadata=True
    )
    return results

def batch_query_pinecone(queries: list[str], 
                        top_k: int = 10, 
                        namespace: str = PINECONE_NAMESPACE, 
                        n_parallel: int = 10) -> list[dict]:
    """Batch query a Pinecone index and return the results.
    
    Internally uses a ThreadPool to parallelize the queries.
    """
    index = get_pinecone_index()
    embeds = batch_embed_queries(queries)
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(n_parallel)
    results = pool.map(
        lambda x: index.query(
            vector=x, 
            top_k=top_k, 
            include_values=False, 
            namespace=namespace, 
            include_metadata=True
        ), 
        embeds
    )
    return results 