#!/bin/bash

# Start script for Rice Disease RAG Service

echo "🚀 Starting Rice Disease RAG Service..."
echo ""

cd /home/likhon/Programming/FYDP/rag_service

# Activate virtual environment and start service
/home/likhon/Programming/FYDP/rag_service/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 4000

echo ""
echo "Service stopped."
