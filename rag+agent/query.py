from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def retriver(query:str)->str:
    question=query
    embedding_model = OllamaEmbeddings(model="nomic-embed-text",base_url="http://localhost:11434")

    vector_store = Chroma(
    collection_name="rag_persistance",
    embedding_function=embedding_model,
    persist_directory="./chromadb"
    )
    # print("Chroma DB loaded successfully")

    retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})

    def structure_docs(retiver):
       return "".join([doc.page_content for doc in retiver])

    prompt=ChatPromptTemplate.from_messages([ (
        "system",
        """You are a helpful RAG assistant.
        Answer the user's question using only the provided context.
        Do not use outside knowledge.
        If the answer cannot be found in the context, say:
        "I don't know based on the provided context."
        Context:{context}"""),
        ("human","{question}")])

    llm=ChatOllama(model="llama3.2",temperature=0)

    rag_chain=({"context":retriever|structure_docs,
            "question":RunnablePassthrough()}
            |prompt|llm|StrOutputParser())

    res=rag_chain.invoke(question)

    return res


