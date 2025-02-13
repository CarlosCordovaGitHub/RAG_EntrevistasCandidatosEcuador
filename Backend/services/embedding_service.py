import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm.notebook import tqdm
from config import Config

class EmbeddingService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {self.device}")
        self.model = SentenceTransformer(Config.MODEL_NAME)
        self.model.to(self.device)

    def encode(self, texts, show_progress=True):
        return self.model.encode(texts, show_progress_bar=show_progress)

    def encode_query(self, query):
        return self.model.encode(query).tolist()

    def process_corpus(self, csv_path, text_processor):
        df = pd.read_csv(csv_path)
        corpus = df[['id', 'candidato_raw', 'temas_tratados_raw', 'descripcion_raw', 'entrevista_raw', 'entrevista_pre']]
        
        corpus['texto_completo'] = corpus['descripcion_raw'] + ' ' + corpus['entrevista_raw']
        corpus['texto_completo_procesado'] = corpus['texto_completo'].apply(text_processor.procesar_texto)
        
        return corpus

    def generate_chunks(self, df_entrevistas, text_processor):
        documents = []
        original_texts = []
        metadata_list = []
        ids = []

        print("Procesando entrevistas...")
        for idx, row in tqdm(df_entrevistas.iterrows(), total=len(df_entrevistas)):
            texto_limpio = text_processor.limpiar_texto(row['texto_completo_procesado'])
            texto_original = row['entrevista_raw']
            
            if texto_limpio:
                chunks = text_processor.chunk_text(texto_limpio)
                chunks_originales = text_processor.chunk_text(texto_original)
                
                for chunk_idx, (chunk, chunk_original) in enumerate(zip(chunks, chunks_originales)):
                    documents.append(chunk)
                    original_texts.append(chunk_original)
                    metadata_list.append({
                        "id_original": str(row["id"]),
                        "candidato": row["candidato_raw"],
                        "chunk_id": chunk_idx,
                        "texto_original": chunk_original
                    })
                    ids.append(f"entrevista_{row['id']}_chunk_{chunk_idx}")

        return documents, original_texts, metadata_list, ids