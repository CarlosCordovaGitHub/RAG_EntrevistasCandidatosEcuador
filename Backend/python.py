# %%
# CELDA 1

import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import ast
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm.notebook import tqdm
import torch

# Descargar stopwords en español (si no lo has hecho previamente)
# nltk.download('stopwords')
# nltk.download('punkt')

# Inicializar el stemmer y obtener las stopwords en español
stemmer = SnowballStemmer('spanish')
stop_words = set(stopwords.words('spanish'))



# %%
df = pd.read_csv('unified_corpus.csv')

# %%
# Filtrar las columnas necesarias (incluir id y candidato)
corpus = df[['id', 'candidato_raw', 'temas_tratados_raw', 'descripcion_raw', 'entrevista_raw', 'entrevista_pre']]

# Limpiar el texto
def limpiar_texto(texto):
    texto = texto.lower()  
    texto = re.sub(r'\d+', '', texto)  
    texto = texto.translate(str.maketrans('', '', string.punctuation))  
    texto = texto.strip()  
    return texto

def procesar_texto(texto):
    texto = limpiar_texto(texto)  
    tokens = word_tokenize(texto, language='spanish')  
    tokens = [token for token in tokens if token not in stop_words]  
    tokens = [stemmer.stem(token) for token in tokens]  
    return ' '.join(tokens)

corpus['texto_completo'] = corpus['descripcion_raw'] + ' ' + corpus['entrevista_raw'] 

corpus['texto_completo_procesado'] = corpus['texto_completo'].apply(procesar_texto)

corpus.to_csv('corpus_procesado_con_texto_completo.csv', index=False)

print("El corpus procesado, incluyendo 'id', 'candidato', y la columna 'texto_completo_procesado', se ha guardado exitosamente en 'corpus_procesado_con_texto_completo.csv'.")



# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# %%
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
model.to(device)

# %%
df = pd.read_csv('corpus_procesado_con_texto_completo.csv')

# %%
print(df.head())

# %%
df_entrevistas = df[['id', 'candidato_raw', 'entrevista_raw', 'texto_completo_procesado']].copy()
print(df_entrevistas.head())

# %%
print("Información del DataFrame de entrevistas:")
print(df_entrevistas.info())
print("\nPrimeras filas:")
print(df_entrevistas.head())

# %%
def chunk_text(text, max_length=512):
    """Divide el texto en chunks más pequeños"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks


# %%
documents = []        # Para los embeddings (texto limpio)
original_texts = []   # Para mostrar (texto original)
metadata_list = []
ids = []


print("Procesando entrevistas...")
for idx, row in tqdm(df_entrevistas.iterrows(), total=len(df_entrevistas)):
    # Usar entrevista_pre para embeddings
    texto_limpio = limpiar_texto(row['texto_completo_procesado'])
    # Guardar entrevista_raw para visualización
    texto_original = row['entrevista_raw']
    
    if texto_limpio:  # Solo procesar si hay texto
        # Dividir en chunks el texto limpio
        chunks = chunk_text(texto_limpio)
        # Dividir en chunks el texto original (misma longitud)
        chunks_originales = chunk_text(texto_original)
        
        # Almacenar cada chunk con sus metadatos
        for chunk_idx, (chunk, chunk_original) in enumerate(zip(chunks, chunks_originales)):
            documents.append(chunk)  # Para embeddings
            original_texts.append(chunk_original)  # Para mostrar
            metadata_list.append({
                "id_original": str(row["id"]),
                "candidato": row["candidato_raw"],
                "chunk_id": chunk_idx,
                "texto_original": chunk_original  # Guardamos el texto original en metadata
            })
            ids.append(f"entrevista_{row['id']}_chunk_{chunk_idx}")


print(f"Total de chunks generados: {len(documents)}")

# %%
print("Generando embeddings...")
embeddings = model.encode(documents, show_progress_bar=True)

# %%
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "entrevistas_candidatos"
collection = chroma_client.get_or_create_collection(name=collection_name)


# %%
batch_size = 100
print("Almacenando en ChromaDB...")
for i in tqdm(range(0, len(documents), batch_size)):
    end_idx = min(i + batch_size, len(documents))
    collection.add(
        documents=documents[i:end_idx],
        embeddings=embeddings[i:end_idx].tolist(),
        metadatas=metadata_list[i:end_idx],
        ids=ids[i:end_idx]
    )

print(f"Almacenamiento completado. Total de chunks: {len(documents)}")

# %%
def search_documents(query, n_results=5):
    """
    Función de búsqueda mejorada que siempre retorna resultados
    """
    collection = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection("entrevistas_candidatos")
    
    # Generar embedding de la query
    query_embedding = model.encode(query).tolist()
    
    # Realizar la búsqueda en ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,  # Directamente pedimos los n_results que queremos
        include=["metadatas", "distances", "documents"]
    )
    
    # Extraer los resultados
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]
    documents = results['documents'][0]
    
    # Crear lista de resultados
    enhanced_results = []
    for i, (distance, metadata, document) in enumerate(zip(distances, metadatas, documents)):
        # Normalizar la similitud para que esté entre 0 y 1
        max_distance = max(distances)
        min_distance = min(distances)
        normalized_similarity = (max_distance - distance) / (max_distance - min_distance) if max_distance != min_distance else 1
        
        enhanced_results.append({
            'metadata': metadata,
            'similarity': normalized_similarity,
            'final_score': normalized_similarity,
            'texto': document
        })
    
    # Ordenar por score final
    enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    return enhanced_results


# %%
def mostrar_resultados(resultados):
    """
    Muestra los resultados de búsqueda de manera clara
    """
    if not resultados:
        print("\nNo se encontraron resultados.")
        return
        
    print("\n=== Resultados de la búsqueda ===\n")
    for i, resultado in enumerate(resultados, 1):
        print(f"\n--- Resultado {i} ---")
        print(f"Candidato: {resultado['metadata']['candidato']}")
        print(f"Relevancia: {resultado['similarity']:.2%}")  
        print("\nTexto:")
        print(f"{resultado['metadata']['texto_original']}")
        print("\n" + "="*50)



# %%
# Probar con varias queries
# Lista para almacenar los resultados
data = []
queries = [
    "¿ cuales son las propuestas para la seguridad?",
]

for query in queries:
    print(f"\nBuscando: '{query}'")
    resultados = search_documents(query, n_results=5)  # Siempre mostrará 3 resultados
    mostrar_resultados(resultados)

# %%
for query in queries:
    print(f"\nBuscando: '{query}'")
    resultados = search_documents(query, n_results=5)

    # Ver la estructura de los resultados
    print("aqui: " + str(resultados))  # <-- Agregar esta línea para depuración

    for res in resultados:
        print(res.keys())  # <-- Ver qué claves tiene cada resultado


# %%
import ollama

def buscar_documentos(query, n_results=5):
    """
    Simula una búsqueda de documentos relevantes.
    Reemplázala con una búsqueda real en tu sistema (por ejemplo, en ChromaDB).
    """
    documentos = [
        {"texto": "El candidato A propone bajar el IVA para mejorar la economía."},
        {"texto": "El candidato B menciona que reducirá impuestos para fomentar el empleo."},
        {"texto": "Candidato C planea eliminar impuestos en ciertos sectores."},
        {"texto": "Candidato D habla sobre políticas fiscales para aumentar la inversión."},
        {"texto": "El candidato E sugiere una reforma fiscal que afectaría a las grandes empresas."}
    ]
    
    return documentos[:n_results]  # Devuelve los primeros 'n_results' documentos

def obtener_resumen_ollama(query, resultados):
    """
    Genera un resumen de los documentos utilizando Ollama.
    """
    textos = [r["texto"] for r in resultados]  # Extraer el texto de cada resultado

    prompt = (
        f"Esto es lo que he encontrado sobre la consulta: {query}\n\n"
        + "\n".join(textos)
        + "\n\nGenera un resumen basado en estos resultados."
    )

    try:
        respuesta = ollama.chat(model='llama3.2:latest', messages=[{'role': 'user', 'content': prompt}])
        return respuesta['message']['content']
    except Exception as e:
        print(f"Error al llamar a Ollama: {str(e)}")
        return None

# 🔹 **Prueba con una consulta**
query_text = "bajar el IVA para fortalecer la economía"

print("\n🔍 Buscando documentos relevantes...")
resultados = buscar_documentos(query_text, n_results=5)  # Simulación de búsqueda

print("\n📑 Generando resumen con Ollama...")
respuesta_ollama = obtener_resumen_ollama(query_text, resultados)

# Imprimir la respuesta generada por Ollama
if respuesta_ollama:
    print("\n📝 Resumen de Ollama:")
    print(respuesta_ollama)


# %%
import ollama
import chromadb

def mostrar_resultados(resultados, query):
    """
    Muestra los resultados de búsqueda de manera clara y genera un resumen con Ollama.
    """
    if not resultados:
        print("\nNo se encontraron resultados.")
        return

    print("\n=== Resultados de la búsqueda ===\n")
    for i, resultado in enumerate(resultados, 1):
        print(f"\n--- Resultado {i} ---")
        print(f"Candidato: {resultado['metadata']['candidato']}")
        print(f"Relevancia: {resultado['similarity']:.2%}")  
        print("\nTexto:")
        print(f"{resultado['metadata']['texto_original']}")
        print("\n" + "="*50)

    # 🔹 Generar resumen con Ollama
    respuesta_ollama = obtener_resumen_ollama(query, resultados)

    if respuesta_ollama:
        print("\n📝 Resumen de Ollama:")
        print(respuesta_ollama)

def obtener_resumen_ollama(query, resultados):
    """
    Genera un resumen estructurado de los documentos utilizando Ollama.
    """
    textos = [f"- {r['metadata']['candidato']}: {r['metadata']['texto_original']}" for r in resultados]

    prompt = (
        f"Consulta: {query}\n\n"
        "He encontrado las siguientes propuestas de los candidatos:\n"
        + "\n".join(textos)
        + "\n\nGenera un resumen estructurado con los siguientes elementos:\n"
        "- Breve resumen individual por candidato.\n"
        "- Una conclusión general sobre los enfoques de los candidatos.\n"
        "Responde de manera clara y concisa."
    )

    try:
        respuesta = ollama.chat(model='llama3.2:latest', messages=[{'role': 'user', 'content': prompt}])
        return respuesta['message']['content']
    except Exception as e:
        print(f"Error al llamar a Ollama: {str(e)}")
        return None


def search_documents(query, n_results=5):
    """
    Función de búsqueda en ChromaDB que retorna documentos relevantes.
    """
    collection = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection("entrevistas_candidatos")

    # Generar embedding de la query
    query_embedding = model.encode(query).tolist()

    # Realizar la búsqueda en ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "distances", "documents"]
    )

    # Extraer los resultados
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]
    documents = results['documents'][0]

    # Crear lista de resultados con similitud normalizada
    enhanced_results = []
    max_distance = max(distances)
    min_distance = min(distances)
    
    for i, (distance, metadata, document) in enumerate(zip(distances, metadatas, documents)):
        normalized_similarity = (max_distance - distance) / (max_distance - min_distance) if max_distance != min_distance else 1
        enhanced_results.append({
            'metadata': metadata,
            'similarity': normalized_similarity,
            'final_score': normalized_similarity,
            'texto': document
        })

    # Ordenar por score final
    enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)

    return enhanced_results

# 🔹 Probar con varias queries
queries = [
    "¿Cuáles son las propuestas para la seguridad?",
    "Sobre que se centra las propuestas de daniel noboa?"
]

for query in queries:
    print(f"\n🔍 Buscando: '{query}'")
    resultados = search_documents(query, n_results=5)  # Buscar en ChromaDB
    mostrar_resultados(resultados, query)  # Mostrar y generar resumen



