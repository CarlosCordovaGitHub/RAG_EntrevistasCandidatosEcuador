import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from config import Config

class TextProcessor:
    def __init__(self):
        self.stemmer = SnowballStemmer('spanish')
        self.stop_words = set(stopwords.words('spanish'))

    def limpiar_texto(self, texto):
        texto = str(texto).lower()
        texto = re.sub(r'\d+', '', texto)
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        texto = texto.strip()
        return texto

    def procesar_texto(self, texto):
        texto = self.limpiar_texto(texto)
        tokens = word_tokenize(texto, language='spanish')
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(tokens)

    def chunk_text(self, text, max_length=512):
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
