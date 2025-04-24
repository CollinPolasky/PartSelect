import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import re
from typing import Dict, List, Optional, Union
import numpy as np

load_dotenv()

def create_searchable_text(row: Dict) -> str:
    """Create rich text representation with multiple search patterns"""
    return f"""
    Product: {row['part_name']}
    Part ID: {row['part_id']}
    MPN: {row['mpn_id']}
    Price: ${row['part_price']}
    Brand: {row['brand']}
    Type: {row['appliance_types']}
    Installation:
    - Difficulty: {row['install_difficulty']}
    - Time: {row['install_time']}
    - Video Guide: {row['install_video_url']}
    Common Symptoms: {row['symptoms']}
    Related Parts: {row['replace_parts']}
    Status: {row['availability']}
    """

def create_part_metadata(row: Dict) -> Dict:
    """Create metadata structure for part entries"""
    return {
        "text": str(row['text']),
        "url": str(row['product_url']),
        "part_id": str(row['part_id']),
        "mpn": str(row['mpn_id']),
        "price": str(row['part_price']),
        "brand": str(row['brand']),
        "appliance_type": str(row['appliance_types']),
        "availability": str(row['availability']),
        "part_name": str(row['part_name']),
        "install_difficulty": str(row['install_difficulty']),
        "install_time": str(row['install_time']),
        "symptoms": str(row['symptoms']),
        "replace_parts": str(row['replace_parts']),
        "install_video_url": str(row['install_video_url'])
    }

class IdentifierExtractor:
    """Extracts various types of identifiers from query text"""
    
    @staticmethod
    def extract_part_id(text: str) -> Optional[str]:
        """Extract PartSelect IDs (PS followed by numbers)"""
        matches = re.findall(r'PS\d+', text, re.IGNORECASE)
        return matches[0].upper() if matches else None
    
    @staticmethod
    def extract_mpn(text: str) -> Optional[str]:
        """Extract Manufacturer Part Numbers (typically alphanumeric)"""
        matches = re.findall(r'(?:MPN|Part Number|Part|#)[\s:]+([A-Z0-9-]+)', text, re.IGNORECASE)
        return matches[0].upper() if matches else None
    
    @staticmethod
    def extract_brand(text: str) -> Optional[str]:
        """Extract brand names"""
        brands = ['whirlpool', 'ge', 'samsung', 'lg', 'maytag']  # Example brands
        words = text.lower().split()
        for brand in brands:
            if brand in words:
                return brand.title()
        return None
    
    @staticmethod
    def extract_appliance_type(text: str) -> Optional[str]:
        """Extract appliance types"""
        types = ['refrigerator', 'dishwasher', 'washer', 'dryer']  # Example types
        words = text.lower().split()
        for type_ in types:
            if type_ in words:
                return type_.title()
        return None
    
    @staticmethod
    def extract_difficulty(text: str) -> Optional[str]:
        """Extract installation difficulty level"""
        difficulties = ['easy', 'moderate', 'difficult', 'professional']
        words = text.lower().split()
        for diff in difficulties:
            if diff in words:
                return diff.title()
        return None
    
    @staticmethod
    def extract_symptoms(text: str) -> Optional[List[str]]:
        """Extract symptoms from text"""
        symptom_indicators = ['not', 'broken', 'leaking', 'noisy', 'won\'t', 'doesn\'t']
        words = text.lower().split()
        found_symptoms = []
        for indicator in symptom_indicators:
            if indicator in words:
                found_symptoms.append(indicator)
        return found_symptoms if found_symptoms else None

def create_search_filters(query: str) -> Dict[str, Union[str, List[str]]]:
    """Create search filters based on identified query parameters"""
    print("\n[Search Strategy] Analyzing query for specific identifiers and filters...")
    extractor = IdentifierExtractor()
    filters = {}
    
    # Try all possible identifiers
    part_id = extractor.extract_part_id(query)
    mpn = extractor.extract_mpn(query)
    brand = extractor.extract_brand(query)
    appliance_type = extractor.extract_appliance_type(query)
    difficulty = extractor.extract_difficulty(query)
    symptoms = extractor.extract_symptoms(query)
    
    # Build filter dictionary and log what was found
    if part_id:
        print(f"[Filter Found] Part ID: {part_id}")
        filters["part_id"] = part_id
    if mpn:
        print(f"[Filter Found] MPN: {mpn}")
        filters["mpn"] = mpn
    if brand:
        print(f"[Filter Found] Brand: {brand}")
        filters["brand"] = brand
    if appliance_type:
        print(f"[Filter Found] Appliance Type: {appliance_type}")
        filters["appliance_type"] = appliance_type
    if difficulty:
        print(f"[Filter Found] Installation Difficulty: {difficulty}")
        filters["install_difficulty"] = difficulty
    if symptoms:
        print(f"[Filter Found] Symptoms: {symptoms}")
        filters["symptoms"] = {"$in": symptoms}
    
    if not filters:
        print("[Search Strategy] No specific filters found - will use semantic search only")
    else:
        print(f"[Search Strategy] Found {len(filters)} filters to narrow search")
    
    return filters

# Initialize Pinecone client and model
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

def vectorize_parts():
    """Vectorize parts data and upload to Pinecone"""
    # Read and prepare parts data
    df = pd.read_csv('case-study/backend/RAG/all_parts.csv')
    
    # Fill NaN values with appropriate defaults
    df = df.fillna({
        'part_name': 'Unknown Part',
        'part_id': 'NO_ID',
        'mpn_id': 'NO_MPN',
        'part_price': '0.00',
        'brand': 'Unknown Brand',
        'appliance_types': 'General Appliance',
        'availability': 'Unknown',
        'product_url': '#',
        'install_difficulty': 'Not Specified',
        'install_time': 'Not Specified',
        'symptoms': 'No symptoms listed',
        'replace_parts': 'No related parts listed',
        'install_video_url': 'No video available'
    })
    
    # Create index if it doesn't exist
    index_name = "parts"
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
    
    # Create searchable text and embeddings
    df['text'] = df.apply(create_searchable_text, axis=1)
    df['embedding'] = df['text'].apply(lambda x: model.encode(x).tolist())
    
    # Prepare vectors for upload
    to_upsert = []
    for i in range(len(df)):
        row = df.iloc[i]
        metadata = create_part_metadata(row)
        
        # Convert embedding to list if it's numpy array
        embedding = row['embedding']
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        to_upsert.append((str(i), embedding, metadata))
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(to_upsert), batch_size):
        batch = to_upsert[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"Uploaded batch {i//batch_size + 1} of {len(to_upsert)//batch_size + 1}")
        except Exception as e:
            print(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
            # Print problematic records for debugging
            for j, (id_, vec, meta) in enumerate(batch):
                try:
                    index.upsert(vectors=[(id_, vec, meta)])
                except Exception as e2:
                    print(f"Problem with record {i+j}: {str(e2)}")
                    print(f"Metadata: {meta}")

def query_parts(query: str, top_k: int = 3) -> Dict:
    """
    Query parts with multi-strategy search
    Handles various types of queries:
    - Part ID/MPN lookups
    - Installation-related queries
    - Symptom-based searches
    - Brand/appliance type filters
    """
    print(f"\n[Query] Processing search: '{query}'")
    
    # Extract search filters
    filters = create_search_filters(query)
    
    # Create vector from query
    print("[Embedding] Creating vector embedding for query...")
    query_vector = model.encode(query).tolist()
    
    index = pc.Index("parts")
    
    # Adjust top_k for symptom searches
    original_top_k = top_k
    if "symptoms" in filters:
        top_k = max(top_k, 5)
        if top_k != original_top_k:
            print(f"[Search Strategy] Increased results count to {top_k} for symptom search")
    
    # If we have filters, try filtered search first
    if filters:
        print("[Search Strategy] Attempting filtered vector search first...")
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filters
        )
        
        if results['matches']:
            print(f"[Results] Found {len(results['matches'])} matches using filters")
            return results
        else:
            print("[Search Strategy] No results found with filters, falling back to semantic search")
    
    # Fall back to semantic search
    print("[Search Strategy] Performing pure semantic search...")
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    print(f"[Results] Found {len(results['matches'])} matches using semantic search")
    return results

# Uncomment to run vectorization
# vectorize_parts()
