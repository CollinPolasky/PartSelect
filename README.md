# PartSelect AI Agent
PartSelect AI Agent is an conversational system designed to assist users with questions about appliance parts, repair guidance, and company policies. The agent uses RAG (Retrieval-Augmented Generation) functions with Pinecone integration to provide accurate, context-aware responses for product, repair, and policy queries.

Highlights
- Smart parts identification product details such as pricing and availability
- Access to comprehensive repair documentation and embedded video guides
- Context-aware conversation handling and memory
- Intelligent RAG retrieval system for accurate responses
- Easy access to company policy information such as return policies and estimated delivery dates


## System Architecture/Design
![Architecture](https://github.com/user-attachments/assets/9f2cd0b4-d68a-4397-82de-de1fe41b7f90)

This flowchart shows the system's architecture including:
- Multi-layered content filters and response validators
- Retrieval process using RAG from three seperate sources (Parts, Repair, and Policy)
- Chat loop flow from query processing and filtering to response generation and validation


# Interface Demo
![PartSelectDemoScreenshotStart](https://github.com/user-attachments/assets/a9a7b120-0cd7-4e6b-b9c7-3fe613a88e20)

![PartSelectDemoScreenshot](https://github.com/user-attachments/assets/bbd9602e-39c9-4b55-a4eb-a93e81d8f2a9)



### Installation
Install the required packages:
pip install -r backend\requirements.txt
npm install

Environment Variables
Create a .env file in the backend directory with the following variables:

- DEEPSEEK_API_KEY=your deepseek api key
- PINECONE_API_KEY=your pinecone api key
- PINECONE_ENVIRONMENT=your pinecone environment/region

Replace "your api key" with your actual API keys.

Note: Make sure to keep your .env file private and never commit it to version control.

# Data
The following files and functions are provided should you wish to replicate the databases for RAG features (in pinecone or otherwise)
- all_parts.csv -> vectorize.py
- repairs.csv -> vectorize_repairs.py
- support_info.json -> vectorize_support.py

Starting the Web Interface

To start the web interface:

npm start

Navigate to the backend directory:

cd backend

Start the server with hot reloading enabled:

uvicorn main:app --reload

Open your web browser and navigate to http://localhost:8000 to access the web interface.
