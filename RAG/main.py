import streamlit as st
from rag import perguntar_politica_RAG

st.write("# ChatBot com IA") # Formato makdown "é editavel". #Titulo
mensagem_usuario = st.chat_input("Escreva sua mensagem aqui") # Input do usuário 

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

