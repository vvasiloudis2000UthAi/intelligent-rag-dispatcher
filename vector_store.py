import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.cluster import KMeans
import numpy as np

load_dotenv()

def create_clusters_and_store(chunks):

    print("Creating embeddings for chunks")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    texts = [chunk.page_content for chunk in chunks]
    vectors = embedding_model.embed_documents(texts)

    print("Clustering embeddings with KMeans")
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(vectors)

    for i, chunk in enumerate(chunks):
        chunk.metadata['cluster'] = int(cluster_labels[i])
        print(f"Chunk {i} assigned to cluster {cluster_labels[i]}")

    print("Storing chunks in Chroma vector store")
    vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db")
    print("Chunks stored successfully in Chroma")
    return vectorstore 