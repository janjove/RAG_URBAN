import re

def chunk_text(text, max_words=300, overlap=50):
    """
    Divideix un text en chunks de mida controlada.
    
    Args:
        text (str): Text netejat.
        max_words (int): Paraules per chunk.
        overlap (int): Paraules que es repeteixen entre chunks.
    Returns:
        list[str]: Llista de chunks.
    """
    # Separa en paraules
    words = re.split(r'\s+', text)
    
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start = end - overlap  # retrocedeix per solapament
    
    return chunks