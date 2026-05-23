import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def read_prompt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def initialize_agent():

    print("Initialize the Topic-Aware RAG Agent")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    llm_router = ChatOpenAI(model="gpt-5o-mini", temperature=0)
    llm_responder = ChatOpenAI(model="gpt-5", temperature=0.2)

    return vectorstore, llm_router, llm_responder

def route_query(query, llm_router):

    raw_prompt = read_prompt_file("prompt/routing_prompt.txt")
    formatted_prompt = raw_prompt.format(query=query)
    response = llm_router(formatted_prompt)
    return response.content.strip()

def execute_rag(query, cluster_id, vectorstore, llm_responder):

    cluster_num = int(cluster_id.split('_')[1])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"cluster": cluster_num}})

    system_prompt_content = read_prompt_file("prompt/system_prompt.txt")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm_responder, prompt_template)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    result = rag_chain.invoke({"input": query})
    return result["answer"]

def run_agent_pipeline(query):

    vectorstore, llm_router, llm_responder = initialize_agent()
    chosen_cluster = route_query(query, llm_router)
    print(f"The router (GPT-5o-mini) assigned the query to cluster: {chosen_cluster}")

    final_answer = execute_rag(query, chosen_cluster, vectorstore, llm_responder)
    return final_answer

