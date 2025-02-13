from flask import Blueprint, request, jsonify
from services.search_service import SearchService
from services.ollama_service import OllamaService
from services.embedding_service import EmbeddingService
from utils.text_processing import TextProcessor

api = Blueprint('api', __name__)
text_processor = TextProcessor()
embedding_service = EmbeddingService()
search_service = SearchService(embedding_service)
ollama_service = OllamaService()

@api.route('/')
def index():
    return jsonify({"message": "API is running"})

@api.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    messages = data.get('messages', [])
    
    if not messages:
        return jsonify({'error': 'Messages are required'}), 400
        
    try:
        response = ollama_service.chat(messages)
        return jsonify({
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')
    n_results = data.get('n_results', 5)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
        
    try:
        results = search_service.search(query, n_results)
        response_data = ollama_service.mostrar_resultados(results, query)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/process-corpus', methods=['POST'])
def process_corpus():
    try:
        data = request.get_json()
        csv_path = data.get('csv_path')
        
        if not csv_path:
            return jsonify({'error': 'CSV path is required'}), 400
            
        corpus = embedding_service.process_corpus(csv_path, text_processor)
        df_entrevistas = corpus[['id', 'candidato_raw', 'entrevista_raw', 'texto_completo_procesado']].copy()
        
        documents, original_texts, metadata_list, ids = embedding_service.generate_chunks(df_entrevistas, text_processor)
        embeddings = embedding_service.encode(documents)
        
        search_service.store_embeddings(documents, embeddings, metadata_list, ids)
        
        return jsonify({'message': 'Corpus processed and stored successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500