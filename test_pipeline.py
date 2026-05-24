import data_pipeline
import vector_store
import agent
import os

def run_pipeline():

    print("Starting the Intelligent RAG Dispatcher Pipeline")
    print("Step 1: Fetching and Processing Documents from arXiv")

    if not os.path.exists("./chroma_db"):
        chunks = data_pipeline.main()

        print("\nStep 2: Clustering and Storing Chunks in Vector Store")
        vector_store.create_clusters_and_store(chunks)
    else:
        print("Chroma vector store already exists. Skipping data fetching and clustering.")
    
    print("\nStep 3: Running the RAG Agent Pipeline")
    print("You can type your questions now. Type 'exit' to quit.")

    while True:
        user_query = input("\nEnter your question:")

        if user_query.strip().lower() in ['exit', 'quit', 'q']:
            print("Exiting the Intelligent RAG Dispatcher Pipeline. Goodbye!")
            break

        if not user_query.strip():
            print("Please enter a valid question.")
            continue

        print("Processing your question...")

        try:
            answer = agent.run_agent_pipeline(user_query)
            print(f"\nAnswer of Agent:\n{answer}")
        except Exception as e:
            print(f"\n An error occurred while processing your question: {e}")

if __name__ == "__main__":
    run_pipeline()
