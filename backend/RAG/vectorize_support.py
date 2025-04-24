import json
import os
from typing import Dict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def create_searchable_text(policy: Dict) -> str:
    """Create rich text representation for policy data"""
    return f"""
    Policy: {policy['title']}
    Content: {policy['content']}
    """

def create_policy_metadata(policy: Dict) -> Dict:
    """Create metadata structure for policy entries"""
    return {
        "title": policy['title'],
        "content": policy['content'],
        "searchable_text": create_searchable_text(policy)
    }

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

def vectorize_support():
    """Vectorize support information and upload to Pinecone"""
    # Read and prepare support data
    with open('case-study/backend/RAG/support_info.json', 'r', encoding='utf-8') as f:
        support_data = json.load(f)
    
    # Create index if it doesn't exist
    index_name = "policy"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    index = pc.Index(index_name)
    
    # Prepare vectors for upload
    to_upsert = []
    for i, policy in enumerate(support_data['policies']):
        # Create metadata
        metadata = create_policy_metadata(policy)
        
        # Create and encode searchable text
        text = metadata['searchable_text']
        embedding = model.encode(text).tolist()
        
        # Generate a unique ID based on the policy title
        policy_id = f"support_{policy['title'].lower().replace(' ', '_')}"
        
        to_upsert.append((policy_id, embedding, metadata))
    
    # Upload in batches
    batch_size = 10
    for i in range(0, len(to_upsert), batch_size):
        batch = to_upsert[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"Uploaded policy batch {i//batch_size + 1} of {len(to_upsert)//batch_size + 1}")
        except Exception as e:
            print(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
            for j, (id_, vec, meta) in enumerate(batch):
                try:
                    index.upsert(vectors=[(id_, vec, meta)])
                except Exception as e2:
                    print(f"Problem with record {i+j}: {str(e2)}")
                    print(f"Metadata: {meta}")

def query_support(query: str, top_k: int = 3) -> Dict:
    """
    Query support information
    Returns dictionary with matches containing metadata and scores
    """
    print(f"\n[Query] Processing support search: '{query}'")
    
    # Create vector from query
    print("[Embedding] Creating vector embedding for support query...")
    query_vector = model.encode(query).tolist()
    
    index = pc.Index("policy")
    
    print("[Search Strategy] Performing semantic support search...")
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )
    
    print(f"[Results] Found {len(results['matches'])} support matches")
    return results


# vectorize_support() 