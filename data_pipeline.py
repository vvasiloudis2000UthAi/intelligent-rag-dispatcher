from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def fetch_arxiv_papers(query, max_results=15):

    print(f"Fetching papers from arXiv with query: '{query}'")
    loader = ArxivLoader(query=query, load_max_docs=max_results)
    docs = loader.load()

    for doc in docs:
        doc.metadata['search_query'] = query
    return docs

def main():
    queries = [
        "Retrieval-Augmented Generation",
        "Convolutional Neural Networks",
        "Generative Adversarial Networks"
    ]

    all_docs = []
    for q in queries:
        papers = fetch_arxiv_papers(q, max_results=20)
        all_docs.extend(papers)

    print(f"Successfully fetched {len(all_docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(all_docs)
    print(f"Total chunks created: {len(chunks)}")

    return chunks

if __name__ == "__main__":
    docs = main()

    print("\nSample first chunk:")
    print(docs[0].page_content[:200] + "...")