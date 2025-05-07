📄 Document Query Bot


A powerful conversational assistant designed for answering document-based queries, retrieving insights from custom PDFs, and generating visualizations on demand. This bot integrates cutting-edge tools like Google Gemini and Pinecone vector databases to deliver fast, intelligent, and context-aware assistance.

---

## 🚀 Features

✅ Conversational Question Answering (Q&A) using advanced language models  
✅ Custom document ingestion from PDFs, searchable through Pinecone  
✅ Visualization generation (e.g., overfitting/underfitting graphs)  
✅ Easy-to-use command-line interface (CLI) for interactive use  
✅ Modular code design for easy extension and customization

---

## 🏗️ Project Structure

document-query-bot/
├── statebot.py # Main script for interactive assistant loop
├── ingestion.py # Script to ingest PDF documents into Pinecone
├── requirements.txt # Python dependencies list
├── .gitignore # Files/directories excluded from Git
├── .env # Environment variables (API keys) [NOT IN GIT]
└── README.md # This project documentation

yaml
Copy
Edit

---

## ⚙️ Requirements

✅ Python 3.8+  
✅ Accounts and API keys:
- Google Gemini (`GEMINI_API_KEY`)
- Pinecone (`PINECONE_API_KEY`)
- Pinecone index (`INDEX_NAME`, `PINECONE_ENVIRONMENT`)

✅ Recommended system tools:
- `git` for version control
- `pip` or `pipenv` for managing Python packages

---

## 📦 Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/document-query-bot.git
cd document-query-bot
Step 2: Set Up a Virtual Environment
Using venv:

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
Step 3: Install Dependencies
bash
Copy
Edit
pip install --upgrade pip
pip install -r requirements.txt
Step 4: Configure Environment Variables
Create a .env file in the root directory with your API keys:

ini
Copy
Edit
GEMINI_API_KEY=your-google-gemini-api-key
PINECONE_API_KEY=your-pinecone-api-key
INDEX_NAME=your-pinecone-index-name
PINECONE_ENVIRONMENT=your-pinecone-environment
✅ Important:

.env is excluded from Git using .gitignore to keep keys secure.

Never share this file publicly!

🏃 Running the Bot
To start querying:

bash
Copy
Edit
python statebot.py
✅ Example questions:

"Summarize the PDF on neural networks."

"What are the key takeaways from the document on climate change?"

"Generate a graph showing overfitting and underfitting."

📄 Document Ingestion
To upload your PDFs into Pinecone for querying:

1️⃣ Place the PDF files in a directory.
2️⃣ Run:

bash
Copy
Edit
python ingestion.py
✅ This script will:

Read and preprocess PDFs.

Split the content into chunks.

Create embeddings.

Upload them to Pinecone.

🧪 Example Use Cases
📚 Researchers: Quickly extract insights from dozens of papers.

🏢 Business Analysts: Query large PDF reports or market documents.

📊 Data Scientists: Generate visual explanations (like loss curves).

🔒 Security Best Practices
✅ Do not commit your .env or API keys to Git.
✅ Regularly rotate your API keys.
✅ Use a virtual environment to isolate dependencies.

🚀 Future Enhancements
✨ Add support for alternative vector stores (e.g., FAISS, Weaviate)
✨ Integrate OpenAI, Anthropic, or other LLM providers
✨ Build a web-based front end with Flask or FastAPI
✨ Add support for image-based or multi-modal queries

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

🤝 Contributing
We welcome contributions!

Fork the repository.

Create a feature branch: git checkout -b feature/your-feature.

Commit your changes.

Push the branch: git push origin feature/your-feature.

Open a pull request.

📬 Contact
For questions or collaboration:

Author: Indrayudh Mukherjee
Email: indrayudh2010@gmail.com
