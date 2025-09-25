import streamlit.cli
import pywebview 
import threading
import sys
import time

# --- Nome do seu arquivo principal do Streamlit ---
streamlit_app_file = "app.py" 
# ---------------------------------------------------

def run_streamlit():
    """Função para iniciar o servidor Streamlit em uma thread."""
    # O comando é equivalente a 'streamlit run app.py --server.headless true'
    # O modo headless é importante para não abrir uma aba de navegador automaticamente
    streamlit.cli.main_run([streamlit_app_file, '--server.headless', 'true', '--server.port', '8501'])

# Inicia o servidor Streamlit em uma thread separada
streamlit_thread = threading.Thread(target=run_streamlit)
streamlit_thread.daemon = True
streamlit_thread.start()

# Aguarda um momento para o servidor Streamlit iniciar
time.sleep(5) 

# Cria a janela do pywebview que aponta para o servidor local do Streamlit
# Verifique se a porta é a mesma que o Streamlit está usando (padrão é 8501)
webview_window = pywebview.create_window(
    'Minha Aplicação Streamlit', 
    'http://localhost:8501',
    width=1200,
    height=800
)

# Inicia o loop de eventos do pywebview
pywebview.start()

# Quando a janela for fechada, o programa principal terminará
# e a thread do Streamlit (daemon) será encerrada junto.
sys.exit()