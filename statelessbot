import os
import sys  # Added for platform detection
import warnings
import time
import subprocess
import tempfile
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from google.generativeai import configure, list_models

warnings.filterwarnings("ignore")
load_dotenv()

console = Console()

MAX_HISTORY_TURNS = 3  # Max Q&A turns to keep in history

class ResearchAssistant:
    def __init__(self, model_preference: str = None):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        configure(api_key=gemini_api_key)

        self.model_name = self._get_best_available_model(user_selected_model=model_preference)
        if not self.model_name:
            raise RuntimeError("No suitable Gemini model found")

        console.print(f"[yellow]Using model: {self.model_name}[/yellow]")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )

        self.vectorstore = PineconeVectorStore(
            index_name=os.getenv("INDEX_NAME"),
            embedding=self.embeddings
        )

        self.chat = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=gemini_api_key,
            temperature=0.3,
            top_k=40,
            top_p=0.95,
            convert_system_message_to_human=True,
        )

        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.chat,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=False,
            verbose=False,
        )
        self.chat_history = []

    def _get_best_available_model(self, user_selected_model: str = None):
        preferred_models = [
            "models/gemini-pro",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-pro-latest",
            "gemini-pro"
        ]

        try:
            available_models = [m.name for m in list_models()]
            for model in preferred_models:
                if model in available_models:
                    return model
            return available_models[0] if available_models else None
        except Exception as e:
            console.print(f"[red]Error checking models: {str(e)}[/red]")
            return "models/gemini-pro"  # Fallback

    def display_answer(self, query, response):
        answer = response["answer"]
        console.print(
            Panel.fit(
                Markdown(f"**Question:** {query}\n\n**Answer:**\n{answer}"),
                title="[bold cyan]dekhlo if it's correct...[/bold cyan]",
                border_style="blue",
            )
        )

    def generate_visual(self, description):
        image_path = os.path.join(tempfile.gettempdir(), "visual_output.png")

        if "overfitting" in description.lower():
            epochs = list(range(1, 21))
            train_loss = [1/(epoch**0.5) for epoch in epochs]
            val_loss = [1/(epoch**0.5) + 0.05*(epoch - 10) if epoch > 10 else 1/(epoch**0.5) for epoch in epochs]

            plt.figure(figsize=(8, 5))
            plt.plot(epochs, train_loss, label='Training Loss')
            plt.plot(epochs, val_loss, label='Validation Loss', linestyle='--')
            plt.axvline(10, color='gray', linestyle=':', label='Overfitting Point')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Overfitting in Machine Learning')
            plt.legend()
            plt.tight_layout()
            plt.savefig(image_path)
            plt.close()
            console.print(f"[green]Overfitting graph saved to:[/green] {image_path}")
            self.open_image(image_path)
        else:
            console.print(f"[red]Sorry, I don't know how to plot that yet![/red]")

    def open_image(self, path):
        try:
            if os.name == 'nt':  # Windows
                os.startfile(path)
            elif os.name == 'posix':  # macOS, Linux
                subprocess.call(['open' if sys.platform == 'darwin' else 'xdg-open', path])
        except Exception as e:
            console.print(f"[red]Could not open image automatically: {str(e)}[/red]")

    def run(self):
        console.print(
            Panel.fit(
                "[bold green]Welcome to nothing![/bold green]\n"
                "I can help answer questions that ain't about yah life.\n"
                "Type 'buhbye', 'exit', or 'quit' to leave.",
                title="[bold]Pucho Pucho...[/bold]"
            )
        )

        while True:
            try:
                query = console.input("\n[bold]Your question:[/bold] ")

                if query.strip().lower() in ('quit', 'exit', 'buhbye'):
                    console.print("\n[bold yellow]Goodbye! See you next time.[/bold yellow]")
                    break

                if not query.strip():
                    continue

                if "graph" in query.lower() or "chart" in query.lower() or "diagram" in query.lower():
                    self.generate_visual(query)
                else:
                    response = self.qa({"question": query, "chat_history": self.chat_history})
                    self.display_answer(query, response)
                    self.chat_history.append((query, response["answer"]))
                    self.chat_history = self.chat_history[-MAX_HISTORY_TURNS:]

            except KeyboardInterrupt:
                console.print("\n\n[bold yellow]Session interrupted. Goodbye![/bold yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    required_vars = ["GEMINI_API_KEY", "INDEX_NAME", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        console.print(
            Panel.fit(
                f"[red]Missing environment variables: {', '.join(missing_vars)}[/red]",
                title="[bold red]Error[/bold red]"
            )
        )
    else:
        try:
            assistant = ResearchAssistant()
            assistant.run()
        except Exception as e:
            console.print(
                Panel.fit(
                    f"[red]Initialization failed: {str(e)}[/red]",
                    title="[bold red]Fatal Error[/bold red]"
                )
            )
