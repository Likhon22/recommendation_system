#!/bin/bash

# Setup script for Rice Disease Detection RAG Service

echo "=============================================="
echo "🌾 Rice Disease RAG Service - Setup"
echo "=============================================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment."
    exit 1
fi

echo ""
echo "✅ Virtual environment created!"
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies."
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ Setup completed successfully!"
echo "=============================================="
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Edit the .env file and add your OpenAI API key:"
echo "   nano .env"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the data ingestion script:"
echo "   python ingest.py"
echo ""
echo "4. Start the service:"
echo "   uvicorn main:app --port 4000"
echo ""
echo "5. Test the API at:"
echo "   http://localhost:4000/docs"
echo ""
