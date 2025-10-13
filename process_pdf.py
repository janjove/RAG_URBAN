from pypdf import PdfReader
import re
import json

def netejar_text(text: str) -> str:
    # 1. Elimina números de pàgina (línies que només tenen digits)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 2. Elimina headers/footers típics (ex: línies repetitives a totes pàgines)
    # → aquí pots afegir patrons concrets del teu document si els detectes
    patrons_headers = [
        r"Confidencial.*",      # Exemple
        r"Nom de l'empresa.*",  # Exemple
    ]
    for p in patrons_headers:
        text = re.sub(p, '', text, flags=re.IGNORECASE)
    
    # 3. Normalitza espais múltiples i línies buides
    text = re.sub(r'\n\s*\n+', '\n\n', text)   # compacta línies buides
    text = re.sub(r'[ \t]+', ' ', text)        # elimina espais múltiples
    
    return text.strip()

def processar_pdf(path_pdf: str, out_json: str = "doc_netejat.json"):
    reader = PdfReader(path_pdf)
    dades = []
    for i, pagina in enumerate(reader.pages, start=1):
        brut = pagina.extract_text() or ""
        net = netejar_text(brut)
        dades.append({
            "pagina": i,
            "text_netejat": net
        })
    
    # Desa com JSON per reutilitzar després
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dades, f, ensure_ascii=False, indent=2)
    print(f"Document netejat desat a {out_json}")


