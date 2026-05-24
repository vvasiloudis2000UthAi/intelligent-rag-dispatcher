import arxiv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def fetch_arxiv_papers(query, max_results=15):

    print(f"Fetching papers from arXiv with query: '{query}'")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    docs = []
    for result in client.results(search):
        text_content = f"Title: {result.title}\n\nAbstract: {result.summary}"
        doc = Document(
            page_content=text_content,
            metadata={
                "search_query": query, 
                "source": result.entry_id,
            }
        )
        docs.append(doc)
        
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