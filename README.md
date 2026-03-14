# SVY Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)

![Telegram](https://img.shields.io/badge/Telegram-26a5e4?style=for-the-badge&logo=telegram)
![LangChain](https://img.shields.io/badge/LangChain-0.3.0-1c1c1c?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

> **AI-Powered Geomatics & Surveying Assistant**
>
> A Retrieval-Augmented Generation (RAG) powered Telegram bot that answers questions related to Geomatics, Land Surveying, GIS, Remote Sensing, and more.

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Deployment](#deployment)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

##  Overview

SVY Agent is an intelligent conversational assistant designed specifically for Surveying and Geoinformatics students and professionals. It leverages state-of-the-art NLP technology to provide accurate, context-aware responses to questions about geomatics topics.

The system uses **Retrieval-Augmented Generation (RAG)** to ensure responses are grounded in authoritative PDF materials, making it ideal for academic and professional use.

##  Features

-  **AI-Powered Conversations** - Natural language responses powered by GPT-4o
-  **PDF Document Processing** - Extracts and processes content from PDF study materials
-  **Semantic Search** - Finds relevant information using embeddings-based retrieval
-  **Telegram Interface** - Easy access via the popular messaging platform
-  **Context-Aware Responses** - Maintains conversation history for coherent dialogue
-  **Debug Mode** - View source documents used for responses

##  Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Telegram   │────▶│  Telegram Bot   │────▶│    RAG Service   │
│   User      │     │   (Python)      │     │   (LangChain)    │
└─────────────┘     └────────┬────────┘     └────────┬─────────┘
                             │                        │
                             ▼                        ▼
                     ┌─────────────────┐     ┌──────────────────┐
                     │  OpenAI GPT-4o │     │    FAISS Index   │
                     │     (LLM)       │     │   (Vector Store) │
                     └─────────────────┘     └──────────────────┘
```

##  Project Structure

```
svy-agent/
├── src/                          # Main source code
│   ├── bot.py                    # Telegram bot entry point (integrated RAG)
│   ├── handlers/                 # Bot command and message handlers
│   │   ├── commands.py           # /start, /help, /status commands
│   │   ├── messages.py           # Message processing logic
│   │   └── errors.py             # Error handling
│   ├── services/                 # Business logic services
│   │   ├── rag_service.py        # RAG pipeline implementation
│   │   └── pdf_processor.py      # PDF extraction and indexing
│   └── utils/                    # Utility modules
│       └── logging_config.py     # Logging configuration
├── config/                       # Configuration management
│   └── __init__.py               # Config dataclass
├── scripts/                      # Utility scripts
│   └── index_documents.py        # Build vector store from PDFs
                 
├── ALL_PDF_FILES/                # Source PDF documents (gitignored)
├── .env.example                  # Environment variables template
├── requirements.txt              # Python dependencies                  
└── README.md                     # This file
```

##  Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| OpenAI API Key | Required |
| Telegram Bot Token | Required |

### Hardware Requirements

| | Minimum | Recommended |
|---|---------|-------------|
| RAM | 8GB | 16GB |
| CPU | 2 cores | 4+ cores |
| Storage | 2GB | 10GB+ (for PDFs) |

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/opeblow/SVY_GPT.git
cd svy-agent
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

##  Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional - RAG Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
CHUNK_SIZE=1200
CHUNK_OVERLAP=100
RETRIEVAL_K=5

# Optional - LLM Configuration
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=500

# Optional - Paths
INDEX_DIR=faiss_index
PDF_DIRECTORY=ALL_PDF_FILES

# Optional
DEBUG=false
```

### Getting Your Telegram Bot Token

1. Open Telegram and search for @BotFather
2. Send `/newbot` to create a new bot
3. Follow the instructions and get your API token
4. Start your bot by sending `/start`

### Getting Your OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Navigate to API Keys
3. Create a new secret key

##  Running the Project

### Step 1: Add PDF Documents

Place your Geomatics-related PDF files in the `ALL_PDF_FILES` directory:

```bash
# Windows
mkdir ALL_PDF_FILES
# Copy your PDFs there

# Or update the path in .env
PDF_DIRECTORY=path/to/your/pdfs
```

### Step 2: Build the Vector Store

```bash
# Activate virtual environment
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Run the indexing script
python run.py index
```

This will:
- Extract text from all PDFs
- Split into chunks
- Create embeddings
- Save the FAISS index

### Step 3: Start the Bot

```bash
# Start the Telegram bot (RAG is initialized automatically)



python run.py bot
```

### Step 4: Test Your Bot

1. Open Telegram and find your bot
2. Send `/start` to see the welcome message
3. Ask a question about Geomatics!

##  Deployment

### Deploying to Render.com (Free)

Render offers a free tier that's perfect for this project.

#### Step 1: Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

#### Step 2: Create Render Account

1. Go to [Render.com](https://render.com)
2. Sign up with GitHub
3. Click "New +" → "Web Service"

#### Step 3: Configure Deployment

| Setting | Value |
|---------|-------|
| Name | svy-agent |
| Region | Oregon (or closest to you) |
| Branch | main |
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python run.py bot` |

#### Step 4: Add Environment Variables

In the Render dashboard, add these environment variables:

- `OPENAI_API_KEY` - Your OpenAI API key
- `TELEGRAM_BOT_TOKEN` - Your Telegram bot token
- `PYTHONUNBUFFERED` - `1`

#### Step 5: Deploy

Click "Create Web Service" and wait for deployment.

**Note:** Since the FAISS index is built locally, you'll need to commit it or use an external storage solution. For production, consider:

1. Building the index locally and committing the `faiss_index` folder
2. Using a cloud storage service (AWS S3, Google Drive)
3. Adding a Google Drive integration 



```bash
# Railway
railway init
railway up

# DigitalOcean App Platform
doctl apps create
```

##  Usage

### Available Commands

| Command | Description |
|---------|-------------|
| `/start` | Start the bot and see welcome message |
| `/help` | Show help information |
| `/status` | Check if backend is running |

### Example Questions

```
What is GIS and its applications in urban planning?
Explain the principles of land surveying
What are the different types of remote sensing?
How does GPS work in surveying?
What is coordinate reference system?
```

### Debug Mode

Set `DEBUG=true` in your `.env` file to see source documents in responses.

##  Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12+ |
| Framework | python-telegram-bot |
| AI/LLM | OpenAI GPT-4o, LangChain |
| Embeddings | HuggingFace sentence-transformers |
| Vector Store | FAISS |
| Deployment | Render.com (free) |

##  Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Contact

For questions or support:
- Email: opeblow2021@gmail.com


---

<div align="center">

Made with ❤️ for Geomatics Students

</div>
