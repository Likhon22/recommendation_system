# 🌾 Rice Disease Detection & Chatbot - RAG Service

A FastAPI-based Retrieval-Augmented Generation (RAG) service that provides multilingual (English/Bangla) chatbot functionality for rice disease detection and treatment advice.

## 📋 Features

- ✅ **Multilingual Support**: Automatically detects and responds in English or Bangla
- ✅ **RAG Pipeline**: Uses LangChain, FAISS, and OpenAI GPT-3.5-turbo
- ✅ **Vector Search**: Efficient similarity search using FAISS
- ✅ **Disease Coverage**: Includes 6 major rice diseases
- ✅ **REST API**: FastAPI with automatic OpenAPI documentation
- ✅ **CORS Enabled**: Ready for frontend integration

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Navigate to the service directory**:

   ```bash
   cd rag_service
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:

   Edit the `.env` file and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

   **How to get your OpenAI API key**:

   - Go to https://platform.openai.com/api-keys
   - Sign in or create an account
   - Click "Create new secret key"
   - Copy the key and paste it in the `.env` file

5. **Add CSV files**:

   Make sure all disease CSV files are in the `rag_service` directory:

   - Bacterial_leaf_blight.csv
   - Brown_Spot_Grain.csv ✅ (already present)
   - Brown_Spot_Leaf.csv
   - Rice_Blast.csv
   - Sheath_blight.csv
   - Sheath_rot.csv

6. **Create the FAISS vector store**:

   ```bash
   python ingest.py
   ```

   This will:

   - Load all CSV files
   - Create embeddings for each Q&A pair
   - Build and save the FAISS index

7. **Start the service**:

   ```bash
   uvicorn main:app --port 4000
   ```

   Or use the shortcut:

   ```bash
   python main.py
   ```

## 📡 API Usage

### Health Check

```bash
curl http://localhost:4000/health
```

### Chat Endpoint (English)

```bash
curl -X POST http://localhost:4000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of brown spot disease?"}'
```

### Chat Endpoint (Bangla)

```bash
curl -X POST http://localhost:4000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "ব্রাউন স্পট রোগের লক্ষণ কি?"}'
```

### Response Format

```json
{
  "answer": "Brown spot disease typically appears as small brown or dark spots...",
  "language": "en",
  "original_query": "What are the symptoms of brown spot disease?",
  "translated_query": null
}
```

### Interactive API Documentation

Visit http://localhost:4000/docs for Swagger UI documentation where you can test all endpoints interactively.

## 📁 Project Structure

```
rag_service/
├── ingest.py                    # Data ingestion script
├── main.py                      # FastAPI application
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys)
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── faiss_index/                 # FAISS vector store (created after ingestion)
├── Bacterial_leaf_blight.csv    # Disease Q&A data
├── Brown_Spot_Grain.csv         # Disease Q&A data
├── Brown_Spot_Leaf.csv          # Disease Q&A data
├── Rice_Blast.csv               # Disease Q&A data
├── Sheath_blight.csv            # Disease Q&A data
└── Sheath_rot.csv               # Disease Q&A data
```

## 🔧 Configuration

### Environment Variables (.env)

| Variable       | Description         | Required |
| -------------- | ------------------- | -------- |
| OPENAI_API_KEY | Your OpenAI API key | Yes      |

### CSV File Format

Each CSV file should have the following columns:

- `Diseases name`: Name of the disease
- `Category`: Category (e.g., Symptoms, Treatment, Prevention)
- `Question`: The question
- `Answer`: The answer

## 🧪 Testing

Test the service using the interactive docs at http://localhost:4000/docs or use curl/Postman.

### Example Test Queries

**English**:

- "What causes brown spot disease?"
- "How do I treat bacterial leaf blight?"
- "What are the symptoms of rice blast?"

**Bangla**:

- "ব্রাউন স্পট রোগের কারণ কী?"
- "ব্যাকটেরিয়াল লিফ ব্লাইট কিভাবে চিকিৎসা করব?"

## 🛠️ Troubleshooting

### Error: "FAISS index not found"

Run `python ingest.py` to create the index first.

### Error: "OPENAI_API_KEY not found"

Make sure you've added your API key to the `.env` file.

### Translation not working

The `googletrans` library may need internet connection. Check your network.

### Missing CSV files

The script will skip missing files with a warning. Add all 6 CSV files for complete coverage.

# 🌾 Rice Disease Detection — RAG Service

A FastAPI-based Retrieval-Augmented Generation (RAG) service that answers questions about rice diseases using a FAISS-backed vector store and a generative LLM. The service supports English and Bangla (Bangla queries are auto-translated).

This README gives step-by-step instructions to set up a Python virtual environment, install dependencies, create the FAISS index from CSV files, run the service locally, and run basic tests.

## Prerequisites

- Python 3.8 or newer
- Bash (instructions assume a Unix-like shell)
- Internet access for model downloads and translations
- An API key for the LLM provider the code uses (see Environment Variables)

## Dependencies

All Python packages are listed in `requirements.txt`. Key packages include:

- fastapi, uvicorn
- langchain, langchain-community, langchain-google-genai
- faiss-cpu, sentence-transformers
- python-dotenv, pandas, deep-translator, langdetect

## Required environment variables

Create a file named `.env` in the `rag_service/` folder (do NOT commit it).

- `GEMINI_API_KEY` — API key for Google Generative AI (the code initializes `ChatGoogleGenerativeAI` and checks this variable).
- Optionally, if you adapt the code to OpenAI, set `OPENAI_API_KEY`.

Example `.env` (DO NOT commit):

```env
GEMINI_API_KEY=your_google_generative_api_key_here
# OPENAI_API_KEY=sk-your-openai-key-if-applicable
```

## Setup (recommended quick sequence)

From the `rag_service` directory:

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Upgrade pip and install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Add your LLM API key(s) to `.env` as shown above.

4. Add or verify that the CSV files used for ingestion are present in `rag_service/`:

- Bacterial_leaf_blight.csv
- Brown_Spot_Grain.csv
- Brown_Spot_Leaf.csv
- Rice_Blast.csv
- Sheath_blight.csv
- Sheath_rot.csv

Each CSV must include the columns: `Diseases name`, `Category`, `Question`, `Answer`.

5. Build the FAISS index by running:

```bash
python ingest.py
```

This creates the directory `faiss_index/` with the saved vector store.

6. Start the service (development):

```bash
uvicorn main:app --host 0.0.0.0 --port 4000
```

Or run directly (same effect):

```bash
python main.py
```

You can use `./setup.sh` to create a venv and install requirements automatically, and `./start.sh` to start the server using `venv`.

## Endpoints & usage

- GET / — service info
- GET /health — health status
- POST /chat — accepts JSON `{ "query": "..." }` and returns a structured response
- GET /stats — vector store stats
- GET /docs — Swagger UI for interactive testing

Example curl call (English):

```bash
curl -X POST http://localhost:4000/chat \
   -H "Content-Type: application/json" \
   -d '{"query": "What are the symptoms of brown spot disease?"}'
```

Bangla example (the service will translate):

```bash
curl -X POST http://localhost:4000/chat \
   -H "Content-Type: application/json" \
   -d '{"query": "ব্রাউন স্পট রোগের লক্ষণ কি?"}'
```

Run the included test harness after the server is up:

```bash
python test_api.py
```

## Troubleshooting

- If `FAISS index not found` error appears: run `python ingest.py` to create `faiss_index/`.
- If LLM initialization fails with missing key: ensure `GEMINI_API_KEY` is in `.env` and loaded (restart the process after changing `.env`).
- If dependencies fail to install: ensure your Python version is >= 3.8 and you have build tools installed (e.g., `build-essential` on Debian/Ubuntu).

## .gitignore guidance

Do NOT commit `.env`, virtual environments, or the FAISS index directory. A recommended `.gitignore` is included in the repository.

## Next steps (optional additions)

- Add a Dockerfile / docker-compose for easier deployment
- Add CI checks and tests for ingestion and API
- Add authentication and CORS restrictions for production

If you want, I can add a Dockerfile or a minimal systemd unit for running the service in production.

---

Thank you — tell me if you'd like a Dockerfile or a `CONTRIBUTING.md` next.
