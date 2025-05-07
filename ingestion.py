import os
import warnings
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from rich.console import Console
from rich.progress import track

# Configuration
warnings.filterwarnings("ignore")
load_dotenv()
console = Console()

class DocumentIngestor:
    def __init__(self, dimension=768):  # Default to 768, can change to 384
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            dimensions=dimension
        )

        self.index_name = os.getenv("INDEX_NAME", f"research-docs-{dimension}d")
        self._setup_index(dimension)

        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def _setup_index(self, dimension):
        if self.index_name not in self.pc.list_indexes().names():
            console.print(f"[yellow]Creating new index '{self.index_name}'[/yellow]")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
            import time
            time.sleep(60)

    def process_pdf(self, file_path: str):
        console.print(f"[blue]Processing: {file_path}[/blue]")
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])

        chunks = self.text_splitter.split_text(text)
        console.print(f"[green]Created {len(chunks)} chunks[/green]")

        documents = [{
            "page_content": chunk,
            "metadata": {
                "source": os.path.basename(file_path),
                "page": i // 3 + 1
            }
        } for i, chunk in enumerate(chunks)]

        return documents

    def ingest_documents(self, file_paths: list):
        all_docs = []
        for file_path in track(file_paths, description="Processing documents..."):
            try:
                all_docs.extend(self.process_pdf(file_path))
            except Exception as e:
                console.print(f"[red]Error processing {file_path}: {str(e)}[/red]")

        if all_docs:
            console.print("[yellow]Storing documents in Pinecone...[/yellow]")
            self.vectorstore.add_documents(all_docs)
            console.print(f"[green]Successfully stored {len(all_docs)} chunks[/green]")
        else:
            console.print("[red]No valid documents to store[/red]")

if __name__ == "__main__":
    required_vars = ["GEMINI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        console.print(f"[red]Missing environment variables: {', '.join(missing_vars)}[/red]")
    else:
        DIMENSION = 768  # or 384 if needed

        ingestor = DocumentIngestor(dimension=DIMENSION)

        # Provide up to 3 PDF paths
        pdf_files = [
            "/Users/indrayudhsmac/Desktop/Top_50_LLM_Interview_Question_1746507060 copy.pdf",
            "/Users/indrayudhsmac/Downloads/conference-template-letter.pdf",
            "/Users/indrayudhsmac/Desktop/Research_Paper_2.pdf"
        ]

        # Check if files exist
        valid_files = [f for f in pdf_files if os.path.exists(f)]
        if not valid_files:
            console.print("[red]No valid PDF files found[/red]")
        else:
            ingestor.ingest_documents(valid_files)
