from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from API_Gemini import api_chave
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from chamada_llm import chamadallm
from typing import Literal, List, Dict

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



#=============================== definindo Prompt ==========================
def prompt():
    prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
    "Você é um Assistente de Políticas Internas (RH/IT) da empresa Matheus Desenvolvimento. "
    "Responda SOMENTE com base no contexto fornecido. "
    "Se não houver base suficiente, responda apenas 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
    ])
    return prompt_rag
prompt_rag = prompt()
llm_triagem = chamadallm()
document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

def chain():
    prompt_rag = prompt()
    llm_triagem = chamadallm()
    document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)
    return document_chain



#====================== CHAMANDO O RAG ===========================
# Formatadores das respostas.
import re, pathlib

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]




#================================================= RAG ==================================================================
# Aqui e chamado o rag, chamando essa função passando o texto/pergunta, será respondido com o que foi ensinado para a IA
def perguntar_politica_RAG(pergunta: str) -> Dict:
  retriever = retrivers()

  docs_relacionados = retriever.invoke(pergunta)

  if not docs_relacionados:
    return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}

  answer = document_chain.invoke({"input": pergunta, "context": docs_relacionados})

  txt = (answer or "").strip()

  if txt.rstrip(".!?") == "Não sei":
    return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}

  return {"answer": txt, "citacoes": formatar_citacoes(docs_relacionados, pergunta), "contexto_encontrado": True}







#========================= teste ==========================
if __name__ == "__main__":
    
    testes = ["Posso reembolsar a internet?",
            "Quero mais 5 dias de trabalho remoto. Como faço?",
            "Posso reembolsar o curso ou trienamento da Alura?",
            "Quantas capivaras tem no Rio Pinheiros?"]

    for msg_teste in testes:
        resposta = perguntar_politica_RAG(msg_teste)
        print(f"PERGUNTA: {msg_teste}")
        print(f"RESPOSTA: {resposta['answer']}")
#        if resposta['contexto_encontrado']:
#            print("CITAÇÕES:")
#            for c in resposta['citacoes']:
#                print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
#                print(f"   Trecho: {c['trecho']}")
        print("------------------------------------")
            

