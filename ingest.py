"""
Data Ingestion Script for Rice Disease RAG System
This script loads CSV files containing Q&A data about rice diseases
and creates a FAISS vector store for efficient retrieval.
"""

import os
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# List of CSV files to ingest
CSV_FILES = [
    "Bacterial_leaf_blight.csv",
    "Brown_Spot_Grain.csv",
    "Brown_Spot_Leaf.csv",
    "Rice_Blast.csv",
    "Sheath_blight.csv",
    "Sheath_rot.csv"
]

def load_csv_data(csv_files):
    """
    Load data from multiple CSV files and convert to LangChain documents.
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        List of Document objects
    """
    documents = []
    
    for csv_file in csv_files:
        file_path = os.path.join(os.path.dirname(__file__), csv_file)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"⚠️  Warning: {csv_file} not found. Skipping...")
            continue
            
        print(f"📄 Loading {csv_file}...")
        
        try:
            # Read CSV file with encoding options
            df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')
            
            # Validate required columns
            required_columns = ['Question', 'Answer', 'Diseases name', 'Category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"⚠️  Warning: {csv_file} missing columns: {missing_columns}. Skipping...")
                continue
            
            # Convert each row to a Document
            for _, row in df.iterrows():
                # Create document content combining question and answer
                content = f"Question: {row['Question']}\nAnswer: {row['Answer']}"
                
                # Create metadata
                metadata = {
                    "disease": row['Diseases name'],
                    "category": row['Category'],
                    "question": row['Question'],
                    "answer": row['Answer'],
                    "source": csv_file
                }
                
                # Create Document object
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            print(f"✅ Loaded {len(df)} documents from {csv_file}")
            
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {str(e)}")
            continue
    
    return documents

def create_vector_store(documents):
    """
    Create FAISS vector store from documents using HuggingFace embeddings.
    
    Args:
        documents: List of Document objects
        
    Returns:
        FAISS vector store object
    """
    print("\n🔧 Initializing HuggingFace embeddings...")
    
    # Initialize embeddings model
    # Using 'all-MiniLM-L6-v2' - a good balance of speed and quality
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("🔨 Creating FAISS vector store...")
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store

def main():
    """
    Main function to orchestrate the ingestion process.
    """
    print("=" * 60)
    print("🌾 Rice Disease Detection RAG System - Data Ingestion")
    print("=" * 60)
    
    # Load documents from CSV files
    print("\n📚 Step 1: Loading CSV files...")
    documents = load_csv_data(CSV_FILES)
    
    if not documents:
        print("\n❌ No documents loaded. Please check your CSV files.")
        return
    
    print(f"\n✅ Total documents loaded: {len(documents)}")
    
    # Create vector store
    print("\n📊 Step 2: Creating vector store...")
    vector_store = create_vector_store(documents)
    
    # Save vector store
    print("\n💾 Step 3: Saving FAISS index...")
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    vector_store.save_local(index_path)
    
    print(f"✅ FAISS index saved to: {index_path}")
    
    print("\n" + "=" * 60)
    print("🎉 Data ingestion completed successfully!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Make sure your .env file has the OPENAI_API_KEY set")
    print("   2. Run: uvicorn main:app --port 4000")
    print("   3. Test the API at: http://localhost:4000/docs")

if __name__ == "__main__":
    main()
