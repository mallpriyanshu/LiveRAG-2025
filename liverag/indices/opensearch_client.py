from functools import cache
import boto3
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

OPENSEARCH_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
AWS_PROFILE_NAME = "sigir-participant"
AWS_REGION_NAME = "us-east-1"

def get_ssm_value(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME) -> str:
    """Get a cleartext value from AWS SSM."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ssm = session.client("ssm")
    return ssm.get_parameter(Name=key)["Parameter"]["Value"]

@cache
def get_client(profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
    """Get an authenticated OpenSearch client."""
    credentials = boto3.Session(profile_name=profile).get_credentials()
    auth = AWSV4SignerAuth(credentials, region=region)
    host_name = get_ssm_value("/opensearch/endpoint", profile=profile, region=region)
    aos_client = OpenSearch(
        hosts=[{"host": host_name, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )
    return aos_client

def query_opensearch(query: str, top_k: int = 1) -> dict:
    """Query an OpenSearch index and return the results."""
    client = get_client()
    results = client.search(
        index=OPENSEARCH_INDEX_NAME, 
        body={
            "query": {
                "match": {
                    "text": query
                }
            }, 
            "size": top_k
        }
    )
    return results

def batch_query_opensearch(queries: list[str], top_k: int = 10, n_parallel: int = 10) -> list[dict]:
    """Sends a list of queries to OpenSearch and returns the results.
    
    Configuration of Connection Timeout might be needed for serving large batches of queries.
    """
    client = get_client()
    request = []
    for query in queries:
        req_head = {"index": OPENSEARCH_INDEX_NAME}
        req_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text"],
                }
            },
            "size": top_k,
        }
        request.extend([req_head, req_body])

    return client.msearch(body=request)

def show_opensearch_results(results: dict):
    """Print OpenSearch results in a readable format."""
    for match in results["hits"]["hits"]:
        print("chunk:", match["_id"], "score:", match["_score"])
        print(match["_source"]["text"])
        print() 