# 🤖 Agente de IA com RAG e Gemini

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)
![LangChain](https://img.shields.io/badge/AI-LangChain-green.svg)
![Gemini](https://img.shields.io/badge/Model-Google%20Gemini-orange.svg)

Este projeto consiste num **Assistente de Políticas Internas** alimentado por Inteligência Artificial. Ele utiliza a técnica de **RAG (Retrieval-Augmented Generation)** para responder a dúvidas de colaboradores sobre normas de RH (Reembolsos, Home Office, Segurança, etc.) com base em documentos PDF oficiais da empresa.

O sistema integra o **Google Gemini** para geração de respostas e embeddings, orquestrado via **LangChain**, com uma interface amigável em **Streamlit**.

---

## 📋 Funcionalidades

* **💬 Chat Interativo:** Interface conversacional simples para envio de perguntas.
* **🔍 Busca Contextual (RAG):** O sistema lê arquivos PDF, divide-os em fragmentos e busca as informações mais relevantes para responder à pergunta do usuário.
* **🧠 Triagem Inteligente:** Um módulo de classificação que identifica a intenção do usuário (Auto Resolver, Pedir Informação ou Abrir Chamado).
* **🛡️ Respostas Fundamentadas:** A IA é instruída a responder apenas com base no contexto fornecido, evitando alucinações sobre políticas inexistentes.
* **📂 Suporte a Múltiplos PDFs:** Carregamento dinâmico de documentos da pasta `docs/`.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM & Embeddings:** Google Gemini (`gemini-2.5-flash`, `gemini-embedding-001`).
* **Orquestração:** LangChain & LangGraph.
* **Vector Store:** FAISS (Facebook AI Similarity Search).
* **Processamento de PDF:** PyMuPDF.

---

## 📂 Estrutura do Projeto

```text
📁 RAG-IA-GEMINI/
├── 📄 main.py                          # Ponto de entrada da aplicação (Interface Streamlit)
├── 📄 import_pdf_splitter_embedding.py # Lógica do RAG (Carregamento, Split, Vector Store)
├── 📄 triagem.py                       # Módulo de classificação de intenção do usuário
├── 📄 API_Gemini.py                    # Configuração de variáveis de ambiente
├── 📄 chamada_llm.py                   # Inicialização do modelo Gemini
├── 📄 flow_langgraph                   # (Opcional) Estrutura avançada de fluxo de decisão
├── 📄 requirements.txt                 # Dependências do projeto
├── 📄 .env                             # Chave da API (Não comitado)
└── 📁 docs/                            # Pasta contendo os PDFs das políticas

```

### Teste do projeto

Projeto funcionando:
   https://conversor-de-pdf-gznaedwrwscwqsxl2flmse.streamlit.app/
