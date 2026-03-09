"""Entry points for running SVY Agent."""
import sys


def run_bot():
    """Run the Telegram bot with integrated RAG."""
    from src.bot import main
    main()


def index_documents():
    """Build the vector store from PDFs."""
    from scripts.index_documents import main
    main()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [bot|index]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "bot":
        run_bot()
    elif command == "index":
        index_documents()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: bot, index")
        sys.exit(1)
