"""
FastAPI-based RAG Service for Rice Disease Detection & Chatbot
This service provides a multilingual (English/Bangla) chatbot interface
for answering questions about rice diseases using RAG (Retrieval-Augmented Generation).
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
from langdetect import detect as detect_language_lib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Rice Disease Detection RAG Service",
    description="Multilingual chatbot for rice disease detection and treatment advice",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG components
vector_store = None
qa_chain = None

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the symptoms of brown spot disease?"
            }
        }

class ChatResponse(BaseModel):
    answer: str
    language: str
    original_query: Optional[str] = None
    translated_query: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Brown spot disease typically appears as small brown or dark spots...",
                "language": "en",
                "original_query": "What are the symptoms of brown spot disease?",
                "translated_query": None
            }
        }

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text: Input text
        
    Returns:
        Language code ('en' for English, 'bn' for Bangla)
    """
    try:
        detected = detect_language_lib(text)
        return detected
    except Exception as e:
        print(f"Language detection error: {e}")
        # Default to English if detection fails
        return 'en'

def translate_text(text: str, dest_lang: str, source_lang: str = 'auto') -> str:
    """
    Translate text to the specified language.
    
    Args:
        text: Text to translate
        dest_lang: Destination language code
        source_lang: Source language code (default: 'auto')
        
    Returns:
        Translated text
    """
    try:
        translator = GoogleTranslator(source=source_lang, target=dest_lang)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def initialize_rag_system():
    """
    Initialize the RAG system with FAISS vector store and QA chain.
    """
    global vector_store, qa_chain
    
    print("🔧 Initializing RAG system...")
    
    # Check if FAISS index exists
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Please run 'python ingest.py' first to create the index."
        )
    
    # Initialize embeddings (must match the one used in ingest.py)
    print("📊 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load FAISS vector store
    print("📚 Loading FAISS vector store...")
    try:
        # Try with the newer parameter first
        vector_store = FAISS.load_local(
            index_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
    except TypeError:
        # Fallback for older versions that don't have this parameter
        vector_store = FAISS.load_local(index_path, embeddings)
    
    # Initialize Google Gemini LLM
    print("🤖 Initializing Google Gemini LLM...")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    # Use the full model resource name as required by the API
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",  
        temperature=0.3,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Create custom prompt template
    prompt_template = """You are an expert agricultural assistant specializing in rice diseases. 
Use the following pieces of context to answer the question about rice diseases. 
If you don't know the answer based on the context, say "I don't have enough information to answer that question accurately."
Always provide detailed, helpful, and accurate information based on the context provided.

Context: {context}

Question: {question}

Answer (provide a detailed, helpful response):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create RetrievalQA chain
    print("⛓️  Creating QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant documents
        ),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("✅ RAG system initialized successfully!")

@app.on_event("startup")
async def startup_event():
    """
    Initialize the RAG system when the application starts.
    """
    try:
        initialize_rag_system()
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("⚠️  The application will start, but chat functionality will not work.")
        print("   Please ensure you have run 'python ingest.py' and set OPENAI_API_KEY in .env")

@app.get("/")
async def root():
    """
    Root endpoint - health check.
    """
    return {
        "service": "Rice Disease Detection RAG Service",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health (GET)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    is_ready = qa_chain is not None and vector_store is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "rag_system_ready": is_ready
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - accepts a query and returns an answer using RAG.
    Supports both English and Bangla languages.
    
    Args:
        request: ChatRequest containing the user's query
        
    Returns:
        ChatResponse with the answer and language information
    """
    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please ensure FAISS index exists and OPENAI_API_KEY is set."
        )
    
    try:
        query = request.query.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Detect language
        detected_lang = detect_language(query)
        print(f"🌐 Detected language: {detected_lang}")
        
        original_query = query
        translated_query = None
        
        # Translate to English if Bangla
        if detected_lang == 'bn':
            print("🔄 Translating Bangla query to English...")
            query = translate_text(query, 'en')
            translated_query = query
            print(f"   Translated: {query}")
        
        # Get answer from RAG chain
        print(f"🔍 Searching for answer...")
        response = qa_chain.invoke({"query": query})
        answer = response['result']
        
        # Translate answer back if original was Bangla
        if detected_lang == 'bn':
            print("🔄 Translating answer to Bangla...")
            answer = translate_text(answer, 'bn')
        
        print("✅ Response generated successfully!")
        
        return ChatResponse(
            answer=answer,
            language=detected_lang,
            original_query=original_query,
            translated_query=translated_query if detected_lang == 'bn' else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing chat request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """
    Get statistics about the vector store.
    """
    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized"
        )
    
    try:
        # Get number of documents in the vector store
        index_size = vector_store.index.ntotal
        
        return {
            "total_documents": index_size,
            "embedding_dimension": vector_store.index.d,
            "status": "operational"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving stats: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
