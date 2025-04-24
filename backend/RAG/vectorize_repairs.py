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
    """Create rich text representation for repair data"""
    return f"""
    Appliance: {row['Product']}
    Problem: {row['symptom']}
    Description: {row['description']}
    Frequency: This issue affects {row['percentage']}% of {row['Product']}s
    Required Parts: {row['parts']}
    Repair Difficulty: {row['difficulty']}
    """

def create_repair_metadata(row: Dict) -> Dict:
    """Create metadata structure for repair entries"""
    # Split parts into a list for better searching
    parts_list = [part.strip() for part in row['parts'].split(',')]
    
    return {
        "appliance_type": str(row['Product']),
        "symptom": str(row['symptom']),
        "description": str(row['description']),
        "frequency": str(row['percentage']),
        "parts_needed": parts_list,
        "symptom_url": str(row['symptom_detail_url']),
        "difficulty": str(row['difficulty']),
        "repair_video": str(row['repair_video_url']),
        "searchable_text": create_searchable_text(row)
    }

class RepairSymptomExtractor:
    """Extracts repair-related information from query text"""
    
    @staticmethod
    def extract_appliance_type(text: str) -> Optional[str]:
        """Extract appliance type"""
        types = ['refrigerator', 'dishwasher', 'washer', 'dryer']
        words = text.lower().split()
        for type_ in types:
            if type_ in words:
                return type_.title()
        return None
    
    @staticmethod
    def extract_symptoms(text: str) -> Optional[List[str]]:
        """Extract common appliance symptoms"""
        symptoms = [
            'noisy', 'leaking', 'not starting', 'not making ice',
            'too warm', 'not dispensing', 'sweating', 'not working',
            'too cold', 'runs too long', 'not cleaning', 'not draining',
            'not filling', 'not dispensing detergent', 'not drying'
        ]
        
        found_symptoms = []
        text_lower = text.lower()
        
        # Check for direct symptom matches
        for symptom in symptoms:
            if symptom in text_lower:
                found_symptoms.append(symptom)
        
        # Check for negative constructions
        negatives = ['won\'t', 'not', 'doesn\'t', 'isn\'t', 'stopped']
        actions = ['start', 'run', 'work', 'clean', 'drain', 'fill', 'dry', 'dispense']
        
        words = text_lower.split()
        for neg in negatives:
            if neg in words:
                idx = words.index(neg)
                if idx + 1 < len(words) and words[idx + 1] in actions:
                    found_symptoms.append(f"not {words[idx + 1]}")
        
        return found_symptoms if found_symptoms else None
    
    @staticmethod
    def extract_difficulty(text: str) -> Optional[str]:
        """Extract repair difficulty level"""
        difficulties = ['REALLY EASY', 'EASY', 'MODERATE', 'DIFFICULT']
        words = text.upper().split()
        for diff in difficulties:
            if diff in words:
                return diff
        return None

def create_repair_filters(query: str) -> Dict[str, Union[str, List[str]]]:
    """Create search filters for repair queries"""
    print("\n[Search Strategy] Analyzing repair query for filters...")
    extractor = RepairSymptomExtractor()
    filters = {}
    
    # Extract search parameters
    appliance_type = extractor.extract_appliance_type(query)
    symptoms = extractor.extract_symptoms(query)
    difficulty = extractor.extract_difficulty(query)
    
    # Build filter dictionary
    if appliance_type:
        print(f"[Filter Found] Appliance Type: {appliance_type}")
        filters["appliance_type"] = appliance_type
    if symptoms:
        print(f"[Filter Found] Symptoms: {symptoms}")
        filters["symptom"] = {"$in": symptoms}
    if difficulty:
        print(f"[Filter Found] Difficulty: {difficulty}")
        filters["difficulty"] = difficulty
    
    if not filters:
        print("[Search Strategy] No specific filters found - will use semantic search only")
    else:
        print(f"[Search Strategy] Found {len(filters)} filters to narrow search")
    
    return filters

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

def vectorize_repairs():
    """Vectorize repair data and upload to Pinecone"""
    # Read and prepare repair data
    df = pd.read_csv('case-study/backend/RAG/repairs.csv')
    
    # Create index if it doesn't exist
    index_name = "repairs"
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
    # to_upsert = []
    # for i, row in df.iterrows():
    #     # Create metadata
    #     metadata = create_repair_metadata(row)
        
    #     # Create and encode searchable text
    #     text = metadata['searchable_text']
    #     embedding = model.encode(text).tolist()
        
    #     to_upsert.append((str(i), embedding, metadata))
    
    # # Upload in batches
    # batch_size = 10  # Smaller batch size for repairs as there are fewer entries
    # for i in range(0, len(to_upsert), batch_size):
    #     batch = to_upsert[i:i+batch_size]
    #     try:
    #         index.upsert(vectors=batch)
    #         print(f"Uploaded repair batch {i//batch_size + 1} of {len(to_upsert)//batch_size + 1}")
    #     except Exception as e:
    #         print(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
    #         for j, (id_, vec, meta) in enumerate(batch):
    #             try:
    #                 index.upsert(vectors=[(id_, vec, meta)])
    #             except Exception as e2:
    #                 print(f"Problem with record {i+j}: {str(e2)}")
    #                 print(f"Metadata: {meta}")

def query_repairs(query: str, top_k: int = 3) -> Dict:
    """
    Query repair information
    Handles various types of queries:
    - Symptom-based searches
    - Appliance-specific searches
    - Difficulty-based filtering
    """
    print(f"\n[Query] Processing repair search: '{query}'")
    
    # Extract search filters
    filters = create_repair_filters(query)
    
    # Create vector from query
    print("[Embedding] Creating vector embedding for repair query...")
    query_vector = model.encode(query).tolist()
    
    index = pc.Index("repairs")
    
    # If we have filters, try filtered search first
    if filters:
        print("[Search Strategy] Attempting filtered repair search...")
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filters
        )
        
        if results['matches']:
            print(f"[Results] Found {len(results['matches'])} repair matches using filters")
            return results
        else:
            print("[Search Strategy] No repair results with filters, falling back to semantic search")
    
    # Fall back to semantic search
    print("[Search Strategy] Performing pure semantic repair search...")
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    print(f"[Results] Found {len(results['matches'])} repair matches using semantic search")
    return results

#vectorize_repairs() 