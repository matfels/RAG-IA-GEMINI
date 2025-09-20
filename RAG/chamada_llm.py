from langchain_google_genai import ChatGoogleGenerativeAI
from API_Gemini import api_chave

chave_api = api_chave()

def chamadallm():
    gemini_flash = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature=0.0,
        api_key= chave_api
    )
    return gemini_flash



if __name__ == "__main__":
    response = chamadallm()
    perg_resp = response.invoke("Estou testando minha API, se você responder é porque funcionou")
    print(perg_resp.content)

