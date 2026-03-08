import os, glob, unicodedata, difflib
import re
from PyPDF2 import PdfReader
import json

# --- CONFIGURATION ---
PDF_TO_SCAN_DIR = "References/" 

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

# --- PARSING & BIBLIOGRAPHIE ---

def parse_reference(ref: str):
    raw = ref.replace("\n", " ").strip()
    # On capture l'année avec suffixe optionnel (ex: 2009a)
    year_m = re.search(r"(19|20)\d{2}[a-z]?", raw)
    year = year_m.group(0) if year_m else ""
    
    authors_part = raw[:year_m.start()] if year_m else raw
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
            if len(tokens) > 1 and len(tokens[0]) == 1 and tokens[0].isalpha():
                tokens[0] = tokens[0] + tokens[1]
                tokens.pop(1)
            surnames.append(norm(tokens[0]))
            for t in tokens[1:]:
                clean_t = re.sub(r"[^a-zA-Z]", "", t)
                if len(clean_t) >= 2 and clean_t.isupper():
                    surnames.append(norm(t))
            
    seen = set(); authors = []
    for s in surnames:
        if s and s not in seen:
            authors.append(s); seen.add(s)
            
    return {"authors": authors, "year": year, "raw": raw}

def extraire_bibliographie_brute(main_pdf_path: str):
    reader = PdfReader(main_pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"
    keywords = [r"références", r"references", r"bibliographie", r"bibliography"]
    pattern = r"\n\s*(" + "|".join(keywords) + r")\s*\n"
    matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
    if not matches: return ""
    return full_text[matches[-1].end():].strip()

def organiser_bibliographie(biblio_brute: str, output_txt: str = "bibliographie_organisee.txt"):
    if not biblio_brute: return []
    text_flux = " ".join(biblio_brute.split())
    pattern_rupture = r"((?:19|20)\d{2}[a-z]?|[0-9]+\s*p\.|[0-9]{2,}\s*[:;]\s*[0-9\-]+)[\.;]?\s+([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ\-']+)"
    text_organise = re.sub(pattern_rupture, r"\1\n\2", text_flux)
    lines = text_organise.split('\n')
    cleaned_entries = []
    for line in lines:
        line = line.strip()
        if len(line) < 25 or "Références" in line: continue
        cleaned_entries.append(line)
    with open(output_txt, "w", encoding="utf-8") as f:
        for entry in cleaned_entries: f.write(entry + "\n")
    return cleaned_entries

# --- LOGIQUE DE SCORE HYBRIDE ---

def calculate_hybrid_score(parsed_ref, pdf_meta, biblio_lines):
    """
    Calcule un score mélangeant la citation directe et la vérité terrain de la biblio.
    """
    # 1. SCORE INITIAL (Veto Auteur/Année)
    score_initial = 0.0
    if parsed_ref["authors"]:
        if any(auth in pdf_meta["norm"] for auth in parsed_ref["authors"]):
            score_initial += 3.0
        else:
            return -50.0 # Veto auteur
    
    if parsed_ref["year"]:
        # Pour le PDF physique, on ignore le suffixe 'a' ou 'b' (souvent absent du nom de fichier)
        ref_year_digits = re.sub(r"[a-z]", "", parsed_ref["year"])
        if pdf_meta["year"] and ref_year_digits != pdf_meta["year"]:
            return -50.0 # Veto année
        if ref_year_digits == pdf_meta["year"]:
            score_initial += 2.0

    # 2. MATCHING AVEC LA LIGNE DE BIBLIO (Pour le titre complet)
    # On cherche dans le TXT la ligne qui contient l'année exacte (ex: 2009a)
    best_bib_line = ""
    max_bib_sim = 0.0
    for line in biblio_lines:
        if parsed_ref["year"] in line: 
            sim = difflib.SequenceMatcher(None, norm(parsed_ref["raw"]), norm(line)).ratio()
            if sim > max_bib_sim:
                max_bib_sim, best_bib_line = sim, line

    # 3. SCORE TITRE (Ligne Biblio vs Nom PDF)
    score_titre = 0.0
    if best_bib_line:
        # C'est ici que 'Effectiveness' vs 'Impact' fait gagner des points
        score_titre = difflib.SequenceMatcher(None, norm(best_bib_line), pdf_meta["norm"]).ratio() * 5.0

    return score_initial + score_titre

def get_smart_context(full_text, match_start, match_end, surrounding_sentences=2):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', full_text)
    current_pos = 0; citation_idx = -1
    for i, sent in enumerate(sentences):
        length = len(sent) + 1
        if current_pos <= match_start < current_pos + length:
            citation_idx = i; break
        current_pos += length
    if citation_idx == -1: return full_text[max(0, match_start-500):min(len(full_text), match_end+500)]
    start = max(0, citation_idx - surrounding_sentences); end = min(len(sentences), citation_idx + surrounding_sentences + 1)
    return " ".join(sentences[start:end])

# --- FONCTION REQUISE PAR L'INTERFACE ---

def build_jobs(main_pdf_path: str, references_dir: str) -> list[dict]:
    if not os.path.exists(references_dir): return []
        
    local_index = list_pdf_files(references_dir)
    # Préparation de la biblio organisée
    brut_text = extraire_bibliographie_brute(main_pdf_path)
    biblio_lines = organiser_bibliographie(brut_text)
    
    doc = PdfReader(main_pdf_path)
    jobs = []
    xp_code_ref = r"~~\s*([a-zA-Z0-9][^~]{1,300})~~"
    compteur = 0

    for i, page in enumerate(doc.pages):
        text_page = " ".join((page.extract_text() or "").split())
        matches = list(re.finditer(xp_code_ref, text_page, re.DOTALL))

        for match in matches:
            compteur += 1
            raw_ref = match.group(1).replace("\n", " ").strip()
            parsed = parse_reference(raw_ref)
            context = get_smart_context(text_page, match.start(), match.end())
            
            best_pdf = None; best_score = -100.0
            
            for pdf_meta in local_index:
                s = calculate_hybrid_score(parsed, pdf_meta, biblio_lines)
                if s > best_score:
                    best_score, best_pdf = s, pdf_meta

            job = {
                "id": f"ref_{compteur}",
                "page": i + 1,
                "raw_citation": raw_ref,
                "citation_context": context,
                "status": "PDF_NOT_FOUND",
                "file_match_score": round(best_score, 2),
            }
            
            # Seuil à 4.0 car on additionne maintenant les scores de base et biblio
            if best_pdf and best_score >= 4.0:
                job.update({
                    "status": "PDF_FOUND",
                    "matched_pdf_path": best_pdf["path"],
                    "pdf_filename": best_pdf["base"]
                })
            jobs.append(job)

    return jobs

if __name__ == "__main__":
    chemin_article = "essai2.pdf"
    if os.path.exists(chemin_article):
        txt_brut = extraire_bibliographie_brute(chemin_article)
        organiser_bibliographie(txt_brut)