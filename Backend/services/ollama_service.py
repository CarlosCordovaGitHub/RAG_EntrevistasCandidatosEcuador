import requests
from config import Config

class OllamaService:
    def __init__(self):
        self.model = Config.OLLAMA_MODEL
        self.base_url = "http://localhost:11434"
        self.session = requests.Session()

    def generate_summary(self, query, results):
        textos = [f"- {r['metadata']['candidato']}: {r['metadata']['texto_original']}" 
                 for r in results]

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
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()['message']['content']
        except Exception as e:
            print(f"Error al llamar a Ollama: {str(e)}")
            return None

    def chat(self, messages):
        """
        Método para chat interactivo con Ollama
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()['message']['content']
        except Exception as e:
            print(f"Error en chat con Ollama: {str(e)}")
            return None

    def mostrar_resultados(self, resultados, query):
        if not resultados:
            return {"message": "No se encontraron resultados."}

        resultados_formateados = []
        for i, resultado in enumerate(resultados, 1):
            resultados_formateados.append({
                "posicion": i,
                "candidato": resultado['metadata']['candidato'],
                "relevancia": f"{resultado['similarity']:.2%}",
                "texto": resultado['metadata']['texto_original']
            })

        resumen = self.generate_summary(query, resultados)

        return {
            "resultados": resultados_formateados,
            "resumen": resumen
        }