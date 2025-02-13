import chromadb
from tqdm.notebook import tqdm
from config import Config

class SearchService:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection("entrevistas_candidatos")

    def store_embeddings(self, documents, embeddings, metadata_list, ids, batch_size=100):
        print("Almacenando en ChromaDB...")
        for i in tqdm(range(0, len(documents), batch_size)):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadata_list[i:end_idx],
                ids=ids[i:end_idx]
            )
        print(f"Almacenamiento completado. Total de chunks: {len(documents)}")

    def search(self, query, n_results=5):
        query_embedding = self.embedding_service.encode_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances", "documents"]
        )
        
        return self._process_results(results)

    def _process_results(self, results):
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        
        enhanced_results = []
        max_distance = max(distances)
        min_distance = min(distances)
        
        for distance, metadata, document in zip(distances, metadatas, documents):
            normalized_similarity = (max_distance - distance) / (max_distance - min_distance) if max_distance != min_distance else 1
            enhanced_results.append({
                'metadata': metadata,
                'similarity': normalized_similarity,
                'final_score': normalized_similarity,
                'texto': document
            })
        
        enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
        return enhanced_results