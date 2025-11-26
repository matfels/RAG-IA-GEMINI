import streamlit as st
#from rag import perguntar_politica_RAG
from import_pdf_splitter_embedding import perguntar_politica_RAG


st.set_page_config(
    page_title="Agente de IA",
    page_icon="🤖",
    layout="wide"
)

st.write("# Agente IA ") # Formato makdown "é editavel". #Titulo
col1, col2 = st.columns([2, 1])
mensagem_usuario = st.chat_input("Escreva sua mensagem aqui") # Input do usuário 
with col1:
        

    # Session_State = Memoria do site, ele funciona como um Dicionário Python.
    if not "lista_mensagem" in st.session_state:
        st.session_state["lista_mensagem"] = []

    # Exibir historico de mensagem do chat
    for mensagem in st.session_state["lista_mensagem"]:
        role = mensagem["role"]
        texto = mensagem["content"]
        st.chat_message(role).write(texto)

    if mensagem_usuario:


        
        # Mensagem humano
        st.chat_message("User").write(mensagem_usuario) # Deixando como chat a conversa
        mensagem = {"role": "user", "content": mensagem_usuario}
        st.session_state["lista_mensagem"].append(mensagem)

        # Resposta IA
        respostaIA = perguntar_politica_RAG(mensagem_usuario)

        # Exibir a resposta da IA na tela 
        st.chat_message("assistant").write(respostaIA['answer']) # Resposta do chat .
        mensagem_IA = {"role": "assistant", "content": respostaIA['answer']}
        st.session_state["lista_mensagem"].append(mensagem_IA)

    
with col2:
    st.header("Como utilizar a aplicação")
    st.markdown("""
    Esta é uma aplicação de chat com um agente de IA treinado para responder perguntas sobre políticas internas do RH de uma empresa.

    **1. Faça uma pergunta:**
    - Digite sua dúvida no campo "Escreva sua mensagem aqui".
    - As perguntas devem ser relacionadas ao assunto disponibilizado pelo RH.
        - Reembolso / Alimentação em viagem / Transporte / Internet para home office / Cursos e certificações / Custos excepcionais
        - Modelo Hibrido / Equipamentos / Segurança / Ergonomia / Conectividade / Solicitação de trabalho remoto.
        - Política de Uso de E-mail e Segurança da Informação
  
                
    **2. Envie e receba a sua resposta:**
    - Pressione Enter para enviar sua pergunta. O agente de IA processará a informação e fornecerá uma resposta baseada nos documentos de política.
    """)
    st.write("---")
    st.subheader("Conecte-se comigo")
    
    

    # Dicionário com seus links de mídia social
    # Substitua '#' pelos seus links reais
    social_media = {
        "LinkedIn": "https://www.linkedin.com/in/matheusferreirademelo/",
        "GitHub": "https://github.com/matfels",
        "Instagram": "https://www.instagram.com/matfels_/?__pwa=1#",
        
    }

    # Criando colunas para os ícones
    cols = st.columns(len(social_media))

    # Ícones (usando emojis como uma alternativa simples)
    # Para ícones reais (SVG/PNG), o processo seria mais complexo
    # e envolveria HTML/CSS com st.markdown(unsafe_allow_html=True)
    icons = {
        "LinkedIn": "🔗",
        "GitHub": "👨‍💻",
        "Instagram": "📸"
    }    
    for index, (platform, link) in enumerate(social_media.items()):
        with cols[index]:
            st.markdown(
                f"[{icons[platform]} {platform}]({link})",
                unsafe_allow_html=True
            )
    st.write("📫", "matheusferreirademelo@outlook.com.br")
   
    
# para fazer rodar # streamlit run RAG/main.py ou streamlit run main.py   no terminal
