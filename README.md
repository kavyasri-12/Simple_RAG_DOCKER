📚 Gemini RAG Document Chatbot

This project is a Streamlit web application that leverages Google's Gemini 1.5 Pro Large Language Model (LLM) for Retrieval-Augmented Generation (RAG). It allows users to upload custom TXT and PDF documents, build a knowledge base from their content, and then ask questions to get answers grounded in the uploaded information. The application is containerized using Docker for easy deployment and portability.

✨ Features

    Document Upload: Easily upload single or multiple TXT (.txt) and PDF (.pdf) files.

    Intelligent Document Processing:

        Extracts text from uploaded documents.

        Splits large texts into manageable, overlapping chunks.

        Generates high-quality embeddings for each chunk using Google's embedding-001 model.

    Efficient Vector Search (FAISS): Utilizes FAISS (Facebook AI Similarity Search) to create a fast and scalable vector index, enabling rapid retrieval of relevant document chunks.

    Context-Aware Generation (Gemini 1.5 Pro):

        Retrieves top-k most relevant document chunks based on user queries.

        Passes the query and retrieved context to Google Gemini 1.5 Pro for accurate and grounded answer generation.

        Explicitly states when an answer cannot be found within the provided context, reducing hallucinations.

    Dynamic Knowledge Base: The application intelligently rebuilds its knowledge base (vector store) when you "Process Selected Documents." This means:

        Adding new documents appends their information.

        Removing documents from the uploader and reprocessing clears their data from the active knowledge base.

    Streamlit UI: An intuitive and interactive web interface built with Streamlit for a smooth user experience.

    Dockerized Deployment: Packaged as a Docker image for consistent and isolated execution across different environments.

🚀 Technologies Used

    Python 3.9+

    Google Generative AI SDK: For interacting with Gemini models.

    Streamlit: For building the interactive web application.

    PyPDF2: For extracting text from PDF documents.

    FAISS: For efficient vector storage and similarity search.

    NumPy: For numerical operations, especially with embeddings.

    python-dotenv: For securely loading API keys from a .env file.

    Docker: For containerization.

⚙️ Getting Started

Follow these instructions to set up and run the application locally.

Prerequisites

    Python 3.9+ installed.

    Docker Desktop (or Docker Engine) installed and running on your system.

    Google Gemini API Key: You need an API key from Google AI Studio.

        Go to Google AI Studio.

        Create a new API key.

1. Clone the Repository

git clone https://github.com/kavyasri-12/Simple_RAG_DOCKER # Replace with your actual repo URL
cd gemini-rag-app

2. Set up Environment Variables

Create a file named .env in the root directory of your project (the same directory as app.py and Dockerfile). Add your Google API key to this file:

GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"

Important: Replace "YOUR_GEMINI_API_KEY_HERE" with your actual API key. Do not commit this file to public repositories.

3. Install Dependencies

Create a requirements.txt file in your project root with the following content:

streamlit
google-generativeai
pypdf2
numpy
faiss-cpu # Or faiss-gpu if you have a compatible GPU setup
python-dotenv

Then install them:

pip install -r requirements.txt

4. Build and Run with Docker (Recommended)

Using Docker ensures a consistent environment and simplifies deployment.

    Build the Docker image:
    Navigate to the project root directory in your terminal and run

docker build -t gemini-rag-app .

This command builds a Docker image named gemini-rag-app from your Dockerfile.

Run the Docker container
    docker run -it --rm -p 8501:8501 gemini-rag-app

        -it: Runs in interactive mode and allocates a pseudo-TTY.

        --rm: Automatically removes the container when it exits.

        -p 8501:8501: Maps port 8501 on your local machine to port 8501 inside the container, where Streamlit runs.

        gemini-rag-app: The name of the Docker image you just built.

    Once the container is running, you will see output in your terminal indicating that Streamlit is serving the app.

5. Run Locally (Alternative)

If you prefer to run the application directly on your local machine without Docker:
Bash

streamlit run app.py

🌐 Usage

    Access the Application:
    Open your web browser and navigate to http://localhost:8501.

    Upload Documents:

        Use the "Upload TXT or PDF files" widget in the sidebar to select one or more .txt or .pdf files.

        You can upload multiple files at once.

    Process Documents:

        After selecting your files, click the "Process Selected Documents" button.

        The application will read, chunk, and embed the content of the uploaded files, building its internal knowledge base. A progress bar will indicate the embedding generation.

        Important: If you remove a file from the uploader and then click "Process Selected Documents" again, the data from the removed file will be deleted from the knowledge base, and only the currently selected files will be processed.

    Ask Questions:

        Once the documents are processed (indicated by a success message), you will see the "Ask a Question" section.

        Type your question into the text area.

        Click the "Get Answer" button.

    View Results:

        The AI's response, based on the content of your documents, will be displayed under "AI Response."

        You can expand the "Source Documents" section to see the exact chunks of text from your uploaded files that the AI used to formulate its answer.

📂 Project Structure

.
├── app.py                  # Main Streamlit application code
├── .env                    # Environment variables (e.g., Google API Key) - KEEP THIS OUT OF VCS!
├── requirements.txt        # Python dependencies
└── Dockerfile              # Docker configuration for building the image

🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, feel free to:

    Fork the repository.

    Create a new branch (git checkout -b feature/your-feature-name).

    Make your changes.

    Commit your changes (git commit -m 'Add new feature').

    Push to the branch (git push origin feature/your-feature-name).

    Open a Pull Request.
