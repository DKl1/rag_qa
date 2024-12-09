# RAG: Retrieval-Augmented Generation System

Welcome to the RAG System, a cutting-edge application that combines powerful retrieval techniques with advanced language models to answer your questions with precision and context. The system seamlessly retrieves relevant documents, refines their relevance, and generates detailed, context-aware responses — complete with citations!

# ✨ Features

	•	Smart Retrieval: Choose from three retrieval methods:
	•	Keyword Search (BM25) for term-based precision.
	•	Semantic Search for meaning-based relevance.
	•	Hybrid Search for the best of both worlds.
	•	Chunking: Breaks long documents into smaller chunks for better search accuracy.
	•	Advanced Reranking: Refines results using a state-of-the-art Cross-Encoder model.
	•	Citations: References retrieved documents in the response for full transparency.
	•	Interactive UI: A sleek and intuitive interface powered by Gradio.
# 🚀 Getting Started


# Prerequisites

Ensure you have Python version **`3.11.00`** installed. You can verify your Python version by running:

```sh
python3.11 --version
```
## Installing pip

To install or upgrade pip, run one of the following commands:

```sh
python3.11 -m pip install --user --upgrade pip
python3.11 -m pip --version
```
or

```sh
pip install --upgrade pip
pip --version
```

# Clone the repo

Clone the repository and install the required dependencies:

```sh
git clone git@github.com:DKl1/rag_qa.git
```
or
```sh
git clone https://github.com/DKl1/rag_qa.git
```
go to the folder
```sh
cd rag_qa
```
## Setting Up the Virtual Environment

### Step 1: Install `virtualenv`

If you don't have `virtualenv` installed, you can install it using pip:

```sh
pip install virtualenv
```

### Step 2: Create a Virtual Environment

Create a virtual environment using the following command:

```sh
python3.11 -m venv env
```

### Step 3: Activate the Virtual Environment

Activate your virtual environment by running the appropriate command based on your operating system:

- **Windows**:

  ```sh
  .\env\Scripts\activate
  ```

- **macOS and Linux**:

  ```sh
  source env/bin/activate
  ```

### Step 4: Install Required Libraries

With your virtual environment activated, install the required libraries listed in the `requirements.txt` file:

```sh
pip install -r requirements.txt
``` 
pip install -r requirements.txt

## Running the Application

To run the application, use one of the following commands

```sh
python app.py
```

or
```sh
gradio app.py
```

## Interact with the System
	•	Enter your OpenAI API Key.
	•	Select a Retrieval Method.
	•	Ask your question, and let the RAG system do the rest!


