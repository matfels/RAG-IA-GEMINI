from import_pdf_splitter_embedding import retrivers, chain
from typing import List, Dict
import re, pathlib


# Formatadores do texto das das respostas.
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




# Aqui e chamado o rag, chamando essa função passando o texto/pergunta, será respondido com o que foi ensinado para a IA
def perguntar_politica_RAG(pergunta: str) -> Dict:
  
  retriever = retrivers()
  document_chain = chain()

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