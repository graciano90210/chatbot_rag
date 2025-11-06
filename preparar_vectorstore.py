import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. Cargar las variables de entorno (tu API key) ---
# Esto carga el archivo .env para que LangChain lo pueda usar
load_dotenv()

# Verifica que la API key se cargó (opcional pero recomendado)
if os.getenv("GOOGLE_API_KEY") is None:
    print("Error: GOOGLE_API_KEY no encontrada.")
    exit()
else:
    print("API Key de Google cargada. Listo para empezar.")

# --- 2. Definir el nombre del PDF y el almacén ---
# Asegúrate que este nombre coincida con tu archivo PDF
pdf_path = "Plan Financiero Detallado.pdf"
# Este es el nombre de la carpeta donde guardaremos nuestro almacén
vectorstore_path = "faiss_index_store"


def crear_vectorstore(pdf_file_path, store_path):
    print(f"Cargando el documento: {pdf_file_path}...")
    # --- 3. Cargar el PDF ---
    # Usamos PyPDFLoader para leer el documento
    loader = PyPDFLoader(pdf_file_path)
    # "load" divide el PDF en una página por documento
    docs = loader.load()

    if not docs:
        print("Error: No se pudo cargar el documento o está vacío.")
        return

    print(f"Documento cargado. {len(docs)} páginas encontradas.")
    print("Troceando el texto...")

    # --- 4. "Trocear" el texto (Chunking) ---
    # Dividimos el texto en pedazos más pequeños
    # chunk_size = 1000 caracteres, chunk_overlap = 100 para no perder contexto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    # "split_documents" aplica el troceado a todas nuestras páginas
    splits = text_splitter.split_documents(docs)

    print(f"Texto troceado en {len(splits)} chunks.")
    print("Creando Embeddings y el Almacén de Vectores (FAISS)...")

    # --- 5. Crear Embeddings ---
    # Usamos el "traductor mágico" de OpenAI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- 6. Crear y Guardar el Almacén de Vectores ---
    # Esto toma todos los "splits", les saca los embeddings (números)
    # y los guarda en un almacén FAISS.
    # "from_documents" hace todo ese trabajo pesado
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # Guardamos el almacén localmente para no tener que repetir esto
    vectorstore.save_local(store_path)

    print(f"¡Listo! Almacén de Vectores guardado en: {store_path}")


# --- Ejecutar la función ---
if __name__ == "__main__":
    crear_vectorstore(pdf_path, vectorstore_path)