import os, glob, unicodedata, difflib
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
    if not os.path.exists(root):
        return []
    files = glob.glob(os.path.join(root, "*.pdf"))
    out = []
    for f in files:
        base = os.path.basename(f)
        n = norm(base)
        # Extraction année
        ym = re.search(r"(19|20)\d{2}", base)
        year = ym.group(0) if ym else ""
        # Extraction premier auteur
        am = re.search(r"^([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ\-']+)", os.path.splitext(base)[0])
        first_author = norm(am.group(1)) if am else ""
        out.append({"path": f, "base": base, "norm": n, "year": year, "first_author": first_author})
    return out

# --- PARSING & SCORING ---

def parse_reference(ref: str):
    raw = ref.replace("\n", " ").strip()
    year_m = re.search(r"(19|20)\d{2}(?!.*(19|20)\d{2})", raw)
    year = year_m.group(0) if year_m else ""
    
    authors_part = raw[:year_m.start()] if year else raw
    authors_part = strip_accents(authors_part)
    authors_part = re.sub(r"\bet\s+(coll|al)\.?\b", "", authors_part, flags=re.I)
    authors_part = re.sub(r"[^a-zA-Z,\-']", " ", authors_part)
    
    chunks = re.split(r",|\bet\b", authors_part, flags=re.I)
    surnames = []
    for ch in chunks:
        ch = ch.strip()
        if not ch: continue
        tokens = ch.split()
        if tokens:
            # --- CORRECTION "L yons" -> "Lyons" ---
            # Si le premier morceau est une lettre isolée (sans point) et qu'il y a une suite
            if len(tokens) > 1 and len(tokens[0]) == 1 and tokens[0].isalpha():
                # On recolle les deux morceaux
                tokens[0] = tokens[0] + tokens[1]
                # On supprime le deuxième morceau qui a été fusionné
                tokens.pop(1)

            surnames.append(norm(tokens[0]))
            # 2. ASTUCE PAHO : On garde aussi les ACRONYMES (mots en MAJUSCULES)
            # Ex: "OrganizationP AHO" -> On garde "AHO"
            for t in tokens[1:]:
                # On nettoie pour vérifier si c'est tout en majuscules (au moins 2 lettres)
                clean_t = re.sub(r"[^a-zA-Z]", "", t)
                if len(clean_t) >= 2 and clean_t.isupper():
                    surnames.append(norm(t))
            
    seen = set(); authors = []
    for s in surnames:
        if s and s not in seen:
            authors.append(s); seen.add(s)
            
    return {"authors": authors, "year": year, "raw": raw}

def score_match(parsed_ref, pdf_meta):
    """
    LOGIQUE ÉQUILIBRÉE :
    1. Auteur OBLIGATOIRE (Sinon -50)
    2. Année : 
       - Si Conflit (2017 vs 2018) -> -50
       - Si Match (2017 vs 2017) -> +2.0
       - Si PDF sans année -> 0 (Neutre, on laisse passer)
    """
    score = 0.0
    
    # --- 1. VETO AUTEUR (OBLIGATOIRE) ---
    if parsed_ref["authors"]:
        found_author = False
        for auth in parsed_ref["authors"]:
            if auth in pdf_meta["norm"]:
                found_author = True
                break
        
        if found_author:
            score += 3.0 # Auteur trouvé -> Base solide
        else:
            return -50.0 # Pas le bon auteur -> Rejet immédiat
            
    # --- 2. LOGIQUE ANNÉE (NUANCÉE) ---
    if parsed_ref["year"]:
        if pdf_meta["year"]:
            # Cas A : Les deux ont une année
            if parsed_ref["year"] == pdf_meta["year"]:
                score += 2.0 # Bonus Match
            else:
                return -50.0 # Conflit -> Rejet (Noel 2018 != 2017)
        else:
            # Cas B : Le PDF n'a PAS d'année
            # Ici, on ne met ni bonus ni malus. 
            # On accepte le risque si l'auteur et le titre correspondent.
            pass 

    # --- 3. SIMILITUDE TITRE ---
    sim = difflib.SequenceMatcher(None, norm(parsed_ref["raw"]), pdf_meta["norm"]).ratio()
    score += 1.0 * sim
    
    return score

def find_best_pdf_for_reference(ref_str, pdf_index):
    p = parse_reference(ref_str)
    
    if not p["authors"] and not p["year"] and len(p["raw"]) < 5:
        return None, 0.0
    
    best = None; best_s = -100.0
    
    for meta in pdf_index:
        s = score_match(p, meta)
        if s > best_s:
            best_s, best = s, meta
            
    # SEUIL : 3.0
    # On met le seuil à 3.0 car :
    # - Si Auteur match (+3.0) + PDF sans année (0) + Similitude (0.5) = 3.5 -> ACCEPTÉ
    # - Si Auteur match (+3.0) + Année match (+2.0) = 5.0 -> ACCEPTÉ
    # - Si Auteur match pas (-50) -> REJETÉ
    # - Si Année conflit (-50) -> REJETÉ
    if best_s >= 3.0:
        return best, best_s
        
    return None, best_s

def get_smart_context(full_text, match_start, match_end, surrounding_sentences=2):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', full_text)
    current_pos = 0
    citation_idx = -1
    
    for i, sent in enumerate(sentences):
        length = len(sent) + 1
        if current_pos <= match_start < current_pos + length:
            citation_idx = i
            break
        current_pos += length
        
    if citation_idx == -1:
        return full_text[max(0, match_start-500):min(len(full_text), match_end+500)]
    
    start = max(0, citation_idx - surrounding_sentences)
    end = min(len(sentences), citation_idx + surrounding_sentences + 1)
    return " ".join(sentences[start:end])

# --- FONCTION REQUISE PAR L'INTERFACE ---

def build_jobs(main_pdf_path: str, references_dir: str) -> list[dict]:
    if not os.path.exists(references_dir):
        return []
        
    local_index = list_pdf_files(references_dir)
    doc = PdfReader(main_pdf_path)
    jobs = []
    xp_code_ref = r"~~\s*([a-zA-Z0-9][^~]{1,300})~~"
    compteur = 0

    for i, page in enumerate(doc.pages):
        text = " ".join((page.extract_text() or "").split())
        matches = list(re.finditer(xp_code_ref, text, re.DOTALL))

        for match in matches:
            compteur += 1
            raw_ref = match.group(1).replace("\n", " ").strip()
            context = get_smart_context(text, match.start(), match.end())
            
            meta, match_score = find_best_pdf_for_reference(raw_ref, local_index)

            job = {
                "id": f"ref_{compteur}",
                "page": i + 1,
                "raw_citation": raw_ref,
                "citation_context": context,
                "status": "PDF_NOT_FOUND",
                "file_match_score": round(match_score, 2),
            }
            
            if meta:
                job["status"] = "PDF_FOUND"
                job["matched_pdf_path"] = meta["path"]
                job["pdf_filename"] = meta["base"]

            jobs.append(job)

    return jobs

if __name__ == "__main__":
    pass