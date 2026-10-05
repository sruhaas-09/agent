
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

loader=PyPDFLoader(file_path="./rag.pdf",mode="page")
documents=loader.load()

chunk_generator=RecursiveCharacterTextSplitter(chunk_size=1800,chunk_overlap=400)
chunks=chunk_generator.split_documents(documents)
print("number of chunks",len(chunks))

embedding_model=OllamaEmbeddings(model="nomic-embed-text")

vector_store=Chroma(collection_name="rag_persistance",
                    embedding_function=embedding_model,
                    persist_directory="./chromadb")

vector_store.add_documents(chunks)

print("vectors and chunks are added to the db sucessfully")