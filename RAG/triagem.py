from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from chamada_llm import chamadallm
from langchain_core.messages import SystemMessage, HumanMessage #Separando a mensagem do sistema e do humano



#Pompt de triagem do Gemini (define como o Gemini deve responder)
def triagem_do_prompt():
    PROMPT = (
        "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
        "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
        "{\n"
        '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
        '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
        '  "campos_faltantes": ["..."]\n'
        "}\n"
        "Regras:\n"
        '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
        '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
        '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
        "Analise a mensagem e decida a ação mais apropriada."
    )
    return PROMPT




#Criação da real triagem do Gemini, limitando a saida do agente. (Define na programação que tera uma saida estruturada. "em listas")
class TriagemOut(BaseModel):
  decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
  urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
  campos_faltantes: List[str] = Field(default_factory=list)






#Chamando o llm e conectando a triagem: 

llm_triagem = chamadallm() #llm
TRIAGEM_PROMPT = triagem_do_prompt() #triagem

triagem_chain = llm_triagem.with_structured_output(TriagemOut)


def triagem(mensagem: str) -> Dict:
  saida: TriagemOut = triagem_chain.invoke([
    SystemMessage(content=TRIAGEM_PROMPT),
    HumanMessage(content=mensagem)
  ])

  return saida.model_dump()  








#======================== TESTE ==============================

if __name__ == "__main__":
    
  testes = ["Posso reembolsar a internet?",
            "Quero mais 5 dias de trabalho remoto. Como faço?",
            "Posso reembolsar o curso ou trienamento da Alura?",
            "Quantas capivaras tem no Rio Pinheiros?"]

  for n in testes:
      print(' ')
      resposta = triagem(n)
      print(f"PERGUNTA: {n}")
      print(f"RESPOSTA: {resposta}")
      #if resposta["contexto_encontrado"]:
      #    print("CITAÇÕES")
      #    print(resposta['citações'])
      print(' ')
      print("-------------------------------==================================-------------------------------")
      
