# Intelligent RAG Dispatcher

## 📌 Project Purpose
The main purpose of the **Intelligent RAG Dispatcher** is to solve the problem of "information noise" and hallucinations that traditional Retrieval-Augmented Generation (RAG) systems exhibit when they are called upon to retrieve data from large and heterogeneous knowledge bases. Instead of a simple, linear keyword search, the Agent introduces a layer of "pre-retrieval intelligence".

Through the use of Semantic Routing and Unsupervised Clustering (K-Means), the system categorizes scientific articles and intelligently routes the user's query to the relevant subset of documents, ensuring high academic validity and strict Guardrails.

---

## 🏗️ System Architecture

The pipeline consists of four main stages:
1. **Data Ingestion (`data_pipeline.py`):** Automatic extraction of scientific abstracts directly from the arXiv API for specific topics (RAG, CNNs, GANs) and slicing them (chunking).
2. **Vector Store & Clustering (`vector_store.py`):** Creation of vector embeddings for each chunk and unsupervised grouping (K-Means clustering) into 3 distinct categories. The data is stored locally in a ChromaDB database.
3. **Semantic Routing (`agent.py`):** A lightweight and fast language model acts as a "Router", analyzing the user's intent and selecting the most appropriate data cluster.
4. **Topic-Aware Generation (`agent.py`):** The primary language model retrieves information *exclusively* from the selected cluster. With strict System Prompts, the model prevents hallucinations, refusing to answer if the information does not exist in the relevant documents.

---

## 🛠️ Technologies & Libraries

The project was developed in Python and utilizes state-of-the-art AI development tools:
* **LangChain (Core, Community, OpenAI, Chroma):** The core framework for Prompt management, Chain creation, and model integration.
* **OpenAI API:** Use of advanced models (`gpt-5-mini` for routing, `gpt-5` for generation, and `text-embedding-3-small` for embeddings).
* **ChromaDB:** Local Vector Database for fast embedding storage and semantic search.
* **Scikit-Learn & NumPy:** For the implementation of the K-Means clustering algorithm.
* **ArXiv API:** For the automated extraction of real scientific articles.
* **Python-dotenv:** For secure API key management.

---

## 🚀 Installation and Execution Instructions

To run this implementation successfully on any machine, please follow these precise steps.

### 1. Prerequisites
Ensure you have Python (>= 3.9) installed on your system. It is highly recommended to use a virtual environment.

### 2. Install Dependencies
Clone the repository and install the required libraries by running the following command in your terminal. We recommend using `python -m pip` to ensure the packages are installed in the correct environment:
```bash
python -m pip install -r requirements.txt
```

*(Note: If you encounter issues with LangChain Chroma during execution, you can run `python -m pip install -U langchain-chroma`)*

### 3. Folder Structure & Crucial Files
Before running the pipeline, your project structure **must** look exactly like this:
```text
intelligent-rag-dispatcher/
│
├── prompts/                  <-- You MUST create this folder
│   ├── routing_prompt.txt    <-- Place your routing prompt here
│   └── system_prompt.txt     <-- Place your system prompt here
│
├── .env                      <-- You MUST create this file for the API key
├── requirements.txt
├── data_pipeline.py
├── vector_store.py
├── agent.py
└── test_pipeline.py
```

### 4. Environment Setup (.env)
You must create a file named exactly **`.env`** (with the dot at the beginning) in the root directory of the project. Open it with a text editor and add your OpenAI API key:
```text
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 5. Running the Pipeline
Once the `.env` file and the `prompts/` folder are set up correctly, you can run the application and start asking the Agent questions:
```bash
python test_pipeline.py
```

* **First execution:** The system will connect to arXiv, download the articles, categorize them, and create the local `chroma_db/` folder. (Duration: ~20-30 seconds).
* **Subsequent executions:** The system recognizes the existing database and jumps straight to the interactive terminal, saving time and API tokens.


## 🧪 QA & Testing Scenarios

During your interaction with the Agent (via the CLI), you can test the robustness of the system with the following scenarios:

* **Routing Test:** Ask specifically about "Convolutional Neural Networks" and observe in the terminal if the Router directs the query to the correct cluster (e.g., `cluster_1`).
* **Anti-Hallucination Test (Guardrails):** Ask a completely off-topic question (e.g., *"What is the best recipe for pasta?"*). The system will refuse to answer, proving that it is strictly limited to its academic scope.
* **Unsupervised Learning Limitations (Label Mismatch):** Because the K-Means algorithm assigns cluster IDs dynamically, a hardcoded routing prompt might fail. In this case, the Agent **will not create a hallucination**, but will recognize that the retrieved documents do not correspond to the question and will safely respond that it lacks the appropriate sources. This proves the system's guardrails are functioning perfectly under edge cases.