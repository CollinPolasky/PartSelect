from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from models import Message, ChatRequest
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import httpx
from collections import defaultdict
from RAG.search_tool import parts_info, repair_info, support_info
from RAG.response_validator import validate_response
import json
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_origin_regex=None,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    expose_headers=[],
    max_age=600,
)

# Initialize clients
async_client = httpx.AsyncClient()
client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=async_client
)

content_filter = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=async_client
)

# Message history
message_history = defaultdict(list)

SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a helpful customer service agent for PartSelect, specializing in Refrigerator and Dishwasher parts. 

    CRITICAL - VIDEO LINKS:
    When providing YouTube video links:
    1. Always provide them on their own line
    2. Use the full URL (https://www.youtube.com/...)
    3. Format: "Watch the installation video here: [URL]"
    4. The video will be automatically embedded in the chat

    You can help with:
    - Finding specific parts by part number
    - Checking part compatibility with specific models
    - Providing installation guidance
    - Troubleshooting appliance issues

    CRITICAL - MEMORY AND CONTEXT:
    1. ALWAYS maintain context from previous messages in the conversation
    2. If a user refers to "that part" or similar, reference the most recently discussed part
    3. Keep track of:
       - Last mentioned part number
       - Last mentioned appliance type
       - Last mentioned brand
       - Last discussed problem/symptom
    4. When answering follow-up questions, explicitly reference the context
       Example: "The Whirlpool Refrigerator Door Shelf Bin (PS11752778) that we discussed costs $36.18"
    5. If context is unclear, reference what you think they're asking about and confirm
       Example: "I believe you're asking about the Whirlpool Door Shelf Bin (PS11752778) we discussed earlier. Is that correct?"

    Please keep responses focused on refrigerator and dishwasher related queries only.
    Your responses should be based in the information provided by the tools and functions at your disposal rather than your own knowledge as much as possible. 
    Do NOT make things up. When in doubt and the tools do not provide a clear answer, say that you do not know.

    When providing product information:
    1. Always begin by checking if there is a tool that can be used to answer the user's query. If there is, use the tool to answer the query.
    2. Always include the part number and price if available
    3. Format installation steps as a numbered list
    4. Include compatibility information when relevant
    5. If multiple parts could be needed, list them all
    6. Always provide a clear call to action (e.g., "Would you like me to provide more details about any of these parts?")
    7. If a tool provides a link in its response, ALWAYS link back the link to the user unless the link is irrelevant to the user's query.

    Important: If a query or answer is about a specific product or part and you've used a tool to find the information about that part, ALWAYS link the product page to the user NEAR THE BEGINNING OF YOUR RESPONSE. It should be: Part Name - Part ID Product URL \n.
    
    Make sure to always maintain a polite, professional, friendly, and helpful tone.
    Do NOT use emojis in your responses.

    Do NOT offer to check or look up the status of an order or package as you are unable to do so. INSTEAD point the user towards the Self Service portal (https://www.partselect.com/user/self-service/) if needed.
    You CAN however, look up the general estimated delivery date policy to provide the user with a generalized idea of when most orders arrive.

    IMPORTANT: Maintain context from previous messages. If a user refers to a previously mentioned part, use that context in your response."""
}

async def check_content(query: str) -> bool:
    """
    Check if the content is appropriate and on-topic.
    Returns True if content is safe and relevant, False otherwise.
    """
    filter_prompt = {
        "role": "system",
        "content": """You are a content filter for an appliance parts customer service system.
        Evaluate if queries are:
        1. On-topic (related to appliance parts, repairs, installation)
        2. Non-malicious (no harmful intent, spam, or inappropriate content)
        3. Safe (no dangerous repair suggestions)
        4. FULLY within scope - Example: Catch and stop any prompt attempting to tack on unrelated parts to an otherwise acceptable query - i.e. "I need a part for my fridge, can you also tell me about the weather in France or write me code?"

        NOTE: Queries that are not actionable and are merely responses such as "Thank you!" or "Perfect!" are allowed and should be responded to a positive confirmation and willingness to help further.
        
        Respond with a confidence score (0-100) and ALLOW/REJECT decision."""
    }
    
    try:
        response = await content_filter.chat.completions.create(
            model="deepseek-chat",
            messages=[
                filter_prompt,
                {"role": "user", "content": f"Query to evaluate: {query}\nProvide score and decision:"}
            ]
        )
        
        result = response.choices[0].message.content.lower()
        # Look for score and decision in the response
        is_allowed = "allow" in result
        score = 0
        try:
            # Try to extract score from response
            score = int(''.join(filter(str.isdigit, result.split()[0])))
        except:
            score = 0 if not is_allowed else 80
            
        print(f"[Content Filter] Score: {score}, Decision: {'ALLOW' if is_allowed else 'REJECT'}")
        return score >= 70 and is_allowed
        
    except Exception as e:
        print(f"[Content Filter] Error: {str(e)}")
        return True  # Default to allowing if filter fails


tools = [
    {
        "type": "function",
        "function": {
            "name": "parts_info",
            "description": """Search for parts information in the PartSelect database - USE THIS if you are trying to find exact information about a part such as price, availability, install time, installation videos, etc. You can also use this to find parts that are compatible with a specific model.
                            if the user is asking for an installation video, link back the video url to the user if available (i.e. youtube link rather than product url)""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for finding parts (can include part numbers, names, brands, etc.)",
                    }
                },
                "required": ["query"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "repair_info",
            "description": """Search for information for repairing problems with appliances in the PartSelect database - USE THIS if you are trying to find solutions to issues that products might be having and to find information such as repair guides, repair videos, parts neeeded, etc. 
                            You can also use this to find parts needed to fix a specific issue.
                            if the user is asking for an repair advice, ALWAYS link back the video url to the user if available (i.e. youtube link rather than product url).
                            Your response should always include the repair video url if available.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for finding repair information (can include parts required, appliance type, symptoms, etc.)",
                    }
                },
                "required": ["query"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "support_info",
            "description": """Search for policy and support information such as return policies, shipping information, warranty details, contact information, etc.
                            USE THIS when users ask about:
                            - Return or refund policies
                            - Shipping policies and tracking
                            - Warranty information
                            - Ordering by phone
                            - Contact information and hours
                            - Estimated delivery dates""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for finding policy and support information",
                    }
                },
                "required": ["query"]
            },
        }
    }
]
@app.post("/reset")
async def reset_chat() -> Message:
    """Reset the chat by clearing message history."""
    message_history.clear()
    return Message(
        role="assistant",
        content="""Welcome to PartSelect's AI Assistant! I specialize in refrigerator and dishwasher parts and am happy to help you with any questions you have on those topics!

I can assist you with:
- Finding specific parts using part numbers or descriptions
- Checking part compatibility with your appliance model
- Providing detailed installation guides and videos
- Troubleshooting appliance problems and estimating repair difficulty and time
- Suggesting related parts you might need
- Explaining repair procedures and requirements
- Answering common questions about PartSelect's policies

All answers come straight from official PartSelect resources, meaning that my responses will be based on official:
- Part specifications and prices
- Installation guides and videos
- Repair documentation
- Compatibility information
- Common problem solutions

How can I help you find the right parts or solve your appliance issues today?"""
    )

@app.post("/chat")
async def chat(request: ChatRequest) -> Message:
    # Get existing conversation/start new one with system prompt
    conversation_id = request.conversation_id if hasattr(request, 'conversation_id') else "default"
    
    if not message_history[conversation_id]:
        message_history[conversation_id] = [SYSTEM_PROMPT]
    
    temp_history = message_history[conversation_id].copy()
    temp_history.append({
        "role": "user",
        "content": request.message
    })
    
    # Run content check and main processing concurrently
    try:
        is_safe, response = await asyncio.gather(
            check_content(request.message),
            client.chat.completions.create(
                model="deepseek-chat",
                messages=temp_history,
                tools=tools
            )
        )
        
        if not is_safe:
            return Message(
                role="assistant",
                content="I apologize, but I can only assist with appliance parts and repair-related questions. Please rephrase your query to focus on these topics."
            )
        
        # If content is safe, update the real message history
        message_history[conversation_id] = temp_history
        assistant_message = response.choices[0].message
        
        # Store search results for validation
        all_search_results = []
        raw_responses = []
        
        # Handle tool calls
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                
                # Execute the appropriate tool and store raw response
                if tool_call.function.name == "parts_info":
                    search_result = parts_info(args["query"])
                    raw_responses.append({
                        "tool": "parts_info",
                        "query": args["query"],
                        "result": search_result
                    })
                elif tool_call.function.name == "repair_info":
                    search_result = repair_info(args["query"])
                    raw_responses.append({
                        "tool": "repair_info",
                        "query": args["query"],
                        "result": search_result
                    })
                elif tool_call.function.name == "support_info":
                    search_result = support_info(args["query"])
                    raw_responses.append({
                        "tool": "support_info",
                        "query": args["query"],
                        "result": search_result
                    })
                else:
                    search_result = f"Error: Unknown tool {tool_call.function.name}"
                    raw_responses.append({
                        "tool": tool_call.function.name,
                        "query": args["query"],
                        "result": search_result
                    })
                
                # Add the tool call to history
                message_history[conversation_id].append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                
                # Add the tool response to history
                message_history[conversation_id].append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": search_result
                })
            
            # Get final response after processing all tool calls
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=message_history[conversation_id]
            )
            assistant_message = response.choices[0].message
            
            # Validate response
            is_satisfactory, analysis, retry_suggestions = await validate_response(
                query=request.message,
                response=assistant_message.content,
                search_results=raw_responses
            )
            
            # If response needs improvement, retry
            if not is_satisfactory and retry_suggestions:
                # Add validation feedback to conversation
                message_history[conversation_id].append({
                    "role": "system",
                    "content": f"Please improve the response. Issues found: {json.dumps(retry_suggestions)}"
                })
                
                # Retry the response
                retry_response = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=message_history[conversation_id]
                )
                assistant_message = retry_response.choices[0].message
        
        # Add assistant's response to history
        message_history[conversation_id].append({
            "role": "assistant",
            "content": assistant_message.content
        })
        
        # Keep only last N messages to prevent context window from growing too large
        if len(message_history[conversation_id]) > 12:  # Adjust this number as needed
            message_history[conversation_id] = [SYSTEM_PROMPT] + message_history[conversation_id][-11:]
        
        return Message(
            role="assistant",
            content=assistant_message.content
        )
    except Exception as e:
        print(f"[Error] Chat processing failed: {str(e)}")
        return Message(
            role="assistant",
            content="I apologize, but I encountered an error processing your request. Please try again."
        ) 