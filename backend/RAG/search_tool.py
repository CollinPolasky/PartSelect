from .vectorize import query_parts
from .vectorize_repairs import query_repairs
from .vectorize_support import query_support

def repair_info(query: str) -> str:
    """Format repair search results into a readable string"""
    try:
        results = query_repairs(query, top_k=3)
        
        if not results['matches']:
            return "No matching parts found."
            
        output = []
        for match in results['matches']:
            metadata = match['metadata']
            output.append(
                f"Problem: {metadata['symptom']}\n"
                f"Appliance: {metadata['appliance_type']}\n"
                f"Description: {metadata['description']}\n"
                f"Frequency: This affects {metadata['frequency']}% of {metadata['appliance_type']}s\n"
                f"Difficulty: {metadata['difficulty']}\n"
                f"Required Parts: {', '.join(metadata['parts_needed'])}\n"
                f"Repair Video: {metadata['repair_video']}\n"
                f"More Info: {metadata['symptom_url']}\n"
                f"Relevance Score: {match['score']:.2f}\n"
                "---"
            )
        
        return "\n".join(output)
    except Exception as e:
        return f"Error searching for parts: {str(e)}"

def parts_info(query: str) -> str:
    """
    Tool for AI to search for parts information
    Returns a formatted string with the search results
    """
    try:
        results = query_parts(query, top_k=3)
        
        if not results['matches']:
            return "No matching parts found."
            
        output = []
        for match in results['matches']:
            metadata = match['metadata']
            # Add installation video URL to output if available
            video_info = f"Installation Video: {metadata['install_video_url']}\n" if metadata.get('install_video_url') and metadata['install_video_url'] != 'No video available' else "Installation Video: Not available\n"
            
            output.append(
                f"Part: {metadata['part_name']}\n"
                f"ID: {metadata['part_id']}\n"
                f"MPN: {metadata['mpn']}\n"
                f"Price: ${metadata['price']}\n"
                f"Brand: {metadata['brand']}\n"
                f"Type: {metadata['appliance_type']}\n"
                f"Installation Difficulty: {metadata['install_difficulty']}\n"
                f"Installation Time: {metadata['install_time']}\n"
                f"Common Symptoms: {metadata['symptoms']}\n"
                f"Related Parts: {metadata['replace_parts']}\n"
                f"Availability: {metadata['availability']}\n"
                f"{video_info}"
                f"URL: {metadata['url']}\n"
                f"Relevance Score: {match['score']:.2f}\n"
                "---"
            )
            
        return "\n".join(output)
        
    except Exception as e:
        return f"Error searching for parts: {str(e)}"

def support_info(query: str) -> str:
    """Format support and policy information search results into a readable string"""
    try:
        results = query_support(query, top_k=2)
        
        if not results['matches']:
            return "No matching information found."
            
        output = []
        for match in results['matches']:
            metadata = match['metadata']
            
            section = [
                f"Policy: {metadata['title']}",
                f"{metadata['content']}\n",
                f"Relevance Score: {match['score']:.2f}",
                "---"
            ]
            
            output.append("\n".join(section))
        
        return "\n".join(output)
    except Exception as e:
        return f"Error searching support information: {str(e)}" 