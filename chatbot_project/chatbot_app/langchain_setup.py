from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


connection = os.getenv("DATABASE_URL")

# Load Hugging Face API token from environment variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    token=HUGGINGFACE_TOKEN
)
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    token=HUGGINGFACE_TOKEN
)

# Initialize text-generation pipeline
text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize PGVector once (shared across app)
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection=connection,
)


def process_pdf(pdf_path):
    """Extracts text, generates embeddings, and stores in PGVector."""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True)
        chunks = text_splitter.split_documents(documents)

        vector_store.add_documents(chunks)  # Store embeddings
        print("PDF processed & stored in PGVector.")
    except Exception as e:
        print(f"Error processing PDF: {e}")


def generate_response(query):
    """Generates AI response using Llama-3.2-3B-Instruct with PDF context."""
    try:
        # Retrieve relevant context from vector database
        results = vector_store.similarity_search(query)
        context_text = "\n".join(
            [r.page_content for r in results]) if results else "No relevant context found."

        # Improved prompt with structured formatting
        prompt = f"""
        You are an AI assistant that answers questions based on provided documents.

        - If relevant context is available, **extract key facts and summarize** them concisely.
        - Use **bullet points or numbered lists** when appropriate.
        - If context does not contain an answer, **summarize what is available** instead of just saying "I don't know."
        - If no relevant data exists, **offer related topics** instead of saying "I don’t know."

        **User Query:** {query}

        **Relevant PDF Context:**
        {context_text}

        **AI Response:**
        """

        # Generate response with Llama-3
        response = text_pipeline(
            prompt, max_new_tokens=150, do_sample=True, temperature=0.3, truncation=True)

        # Extract only AI response
        # Extract only AI response and format properly
        if "AI Response:" in response[0]['generated_text']:
            ai_response = response[0]['generated_text'].split(
                "AI Response:")[-1].strip()

            # Convert bullet points to proper formatting
            # Convert bullets to markdown-style dashes
            ai_response = ai_response.replace("•", "\n-")
            # Remove any unwanted asterisks
            ai_response = ai_response.replace("**", "")

        else:
            ai_response = "Sorry, I could not extract a valid response."

        return ai_response  # Only return the AI-generated text

    except Exception as e:
        return f"Error generating response: {e}"
