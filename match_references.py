import os, glob, unicodedata, difflib, subprocess, sys
import re
from PyPDF2 import PdfReader
import json

# --- CONFIGURATION ---
PDF_TO_SCAN_DIR = "References/"
PDF_INDEX = [] 

# --- FONCTIONS UTILITAIRES ---

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm(s: str) -> str:
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def list_pdf_files(root: str):
    # Liste tous les PDF du dossier References/
    files = glob.glob(os.path.join(root, "*.pdf"))
    out = []
    for f in files:
        base = os.path.basename(f)
        n = norm(base)
        ym = re.search(r"(19|20)\d{2}", base)
        year = ym.group(0) if ym else ""
        am = re.search(r"^([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ\-']+)", os.path.splitext(base)[0])
        first_author = norm(am.group(1)) if am else ""
        out.append({"path": f, "base": base, "norm": n, "year": year, "first_author": first_author})
    return out

# On initialise l'index des PDFs disponibles
PDF_INDEX = list_pdf_files(PDF_TO_SCAN_DIR)

# --- PARSING & SCORING ---

def parse_reference(ref: str):
    raw = ref.replace("\n", " ").strip()
    year_m = re.search(r"(19|20)\d{2}(?!.*(19|20)\d{2})", raw)
    year = year_m.group(0) if year_m else ""
    authors_part = raw[:year_m.start()] if year else raw
    
    authors_part = re.sub(r"\bet\s+(coll|al)\.?\b", "", authors_part, flags=re.I)
    authors_part = re.sub(r"[^a-zA-Z,\-']", " ", authors_part)
    
    chunks = re.split(r",|\bet\b", authors_part, flags=re.I)
    surnames = []
    for ch in chunks:
        ch = ch.strip()
        if not ch: continue
        tokens = ch.split()
        if tokens:
            surnames.append(norm(tokens[0]))
            
    seen = set(); authors = []
    for s in surnames:
        if s and s not in seen:
            authors.append(s); seen.add(s)
            
    return {"authors": authors, "year": year, "raw": raw}

def score_match(parsed_ref, pdf_meta):
    score = 0.0
    if parsed_ref["year"] and parsed_ref["year"] == pdf_meta["year"]:
        score += 2.0
    
    hits = 0
    if parsed_ref["authors"]:
        if parsed_ref["authors"][0] in pdf_meta["norm"]:
            score += 1.5
            hits += 1
        for auth in parsed_ref["authors"][1:]:
            if auth in pdf_meta["norm"]:
                score += 1.0
                hits += 1
                
    if hits >= 2: score += 0.5
    
    sim = difflib.SequenceMatcher(None, norm(parsed_ref["raw"]), pdf_meta["norm"]).ratio()
    score += 0.5 * sim
    return score

def find_best_pdf_for_reference(ref_str, pdf_index=PDF_INDEX):
    p = parse_reference(ref_str)
    if not p["authors"] and not p["year"]:
        return None, 0.0
    
    best = None; best_s = -1.0
    for meta in pdf_index:
        s = score_match(p, meta)
        if s > best_s:
            best_s, best = s, meta
            
    if best_s >= 2.2:
        return best, best_s
    return None, best_s

# --- LE CŒUR DE LA MODIFICATION : CONTEXTE INTELLIGENT ---

def get_smart_context(full_text, match_start, match_end, surrounding_sentences=2):
    """ 
    Récupère la phrase complète de la citation + 2 phrases avant et après.
    Évite de couper au milieu d'une idée.
    """
    # Découpage intelligent des phrases (sur . ? ! suivis d'espace)
    # Le regex complexe évite de couper sur "M. Dupont" ou "Fig. 1"
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', full_text)
    
    current_pos = 0
    citation_idx = -1
    
    # On trouve dans quelle phrase se situe la citation
    for i, sent in enumerate(sentences):
        length = len(sent) + 1 # +1 pour l'espace consommé par le split
        if current_pos <= match_start < current_pos + length:
            citation_idx = i
            break
        current_pos += length
        
    if citation_idx == -1:
        # Fallback de sécurité si on ne trouve pas
        return full_text[max(0, match_start-500):min(len(full_text), match_end+500)]
    
    # On prend n phrases avant et après
    start = max(0, citation_idx - surrounding_sentences)
    end = min(len(sentences), citation_idx + surrounding_sentences + 1)
    
    return " ".join(sentences[start:end])

def create_verification_jobs(doc, output_json="verification_jobs.json"):
    jobs = []
    xp_code_ref = r"~~(.*?)~~"
    compteur = 0
    
    print("Scan des références et création des tâches avec contexte intelligent...")
    
    for i, page in enumerate(doc.pages):
        text = page.extract_text() or ""
        matches = list(re.finditer(xp_code_ref, text, re.DOTALL))
        
        for match in matches:
            compteur += 1
            raw_ref = match.group(1).replace("\n", " ").strip()
            
            # APPEL DE LA NOUVELLE FONCTION
            context = get_smart_context(text, match.start(), match.end())
            
            meta, match_score = find_best_pdf_for_reference(raw_ref)
            
            job = {
                "id": f"ref_{compteur}",
                "page": i + 1,
                "raw_citation": raw_ref,
                "citation_context": context,
                "status": "PDF_NOT_FOUND",
                "file_match_score": round(match_score, 2)
            }
            
            if meta:
                job["status"] = "PDF_FOUND"
                job["matched_pdf_path"] = meta["path"]
                job["pdf_filename"] = meta["base"]
                
            jobs.append(job)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=4, ensure_ascii=False)
    print(f"Terminé. {len(jobs)} tâches enregistrées dans {output_json}")

# --- LANCEMENT ---

if __name__ == "__main__":
    pdf_path = "./essai.pdf" # Vérifiez que ce fichier existe bien à côté du script
    if os.path.exists(pdf_path):
        doc = PdfReader(pdf_path)
        create_verification_jobs(doc)
    else:
        print(f"Erreur : Le fichier {pdf_path} est introuvable.")