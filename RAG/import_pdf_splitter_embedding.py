from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from API_Gemini import api_chave


docs = []


#=================== PDF ===================
#importação dos PDFs
def pdf():
    for n in Path("docs").glob('*.pdf'):
        try:
            loader = PyMuPDFLoader(str(n))
            
            docs.extend(loader.load())
            #print(f"Carregado arquivo com sucesso: {n.name}")
        except Exception as e:
            print(f"Erro ao carregar o arquivo: {n.name}: {e}")

    print('========================')
    print('')
    print(f"Total de documentos 'RAG' carregados: {len(docs)}")
    print('')
    print('========================')

    return docs






#=================== Splitts ===================
#Realizando os Splitts dos PDF. "Particionando eles em pedaços menores para melhorar o desemprenho do sistema".
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=10,
#    length_function=len
)

chunks = splitter.split_documents(pdf())





#=================== embedding =================== 
def retrivers():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key= api_chave()
    )


    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", 
                                        search_kwargs={"score_threshold": 0.3, "k": 4})
    return retriever




#if __name__ == "__main__":
#    for chunk in chunks:
#        print(chunk.page_content+"\n----------------")
    

