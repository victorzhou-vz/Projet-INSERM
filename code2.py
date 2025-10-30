import os, glob, unicodedata, difflib, subprocess, sys
import re
from PyPDF2 import PdfReader

PDF_TO_SCAN_DIR = "References/"          # dossier où se trouvent les PDFs des références
OPEN_MATCHED_PDFS = False      # passe à True si tu veux ouvrir automatiquement

pdf_path = "./essai.pdf"
doc = PdfReader(pdf_path)

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm(s: str) -> str:
    s = strip_accents(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def TrouveReference(text):
    xp_code_ref = r"~~(.*?)~~"
    return re.findall(xp_code_ref, text, re.DOTALL)

# --- Parsing des références ---------------------------------------------------

def parse_reference(ref: str):
    """
    Exemples d'entrées :
      "Holland M, 2014"
      "Chalfin A, 2015"
      "Mao L-Z, Zhu H-G, Duan L-R, 2012"
      "Markandya et Pearce, 1989"
      "Rehm et coll., 2014"
      "Kopp, 2015"
      "Kopp et Ogrodnik, 2017"
    Retourne dict: {'authors': ['holland'], 'year': '2014'}
    """
    raw = ref.replace("\n", " ").strip()
    # isole l'année (la dernière année mentionnée)
    year_m = re.search(r"(19|20)\d{2}(?!.*(19|20)\d{2})", raw)
    year = year_m.group(0) if year_m else ""
    authors_part = raw
    if year:
        authors_part = raw[:year_m.start()]

    # nettoie le segment auteurs
    authors_part = re.sub(r"\bet\s+coll\.?\b", "", authors_part, flags=re.I) # retire "et coll."
    authors_part = re.sub(r"\bet\s+al\.?\b", "", authors_part, flags=re.I)   # retire "et al."
    authors_part = authors_part.replace("’", "'")
    authors_part = re.sub(r"\s*,\s*$", "", authors_part)  # virgule finale

    # sépare sur virgules ou " et "
    chunks = re.split(r",|\bet\b", authors_part, flags=re.I)
    surnames = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        # prends le premier mot “type nom de famille”
        # ex: "Mao L-Z" -> "Mao" ; "Bonaldi C" -> "Bonaldi"
        # ex: "Kopp P-A" -> "Kopp"
        m = re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ\-']+", ch)
        if m:
            surnames.append(norm(m.group(0)))
    # unique et conserve l’ordre
    seen = set(); authors = []
    for s in surnames:
        if s and s not in seen:
            authors.append(s); seen.add(s)

    return {"authors": authors, "year": year, "raw": raw}

# --- Index des fichiers PDF ---------------------------------------------------

def list_pdf_files(root: str):
    files = glob.glob(os.path.join(root, "*.pdf"))
    out = []
    for f in files:
        base = os.path.basename(f)
        n = norm(base)
        # essaie d'attraper une année dans le nom du fichier
        ym = re.search(r"(19|20)\d{2}", base)
        year = ym.group(0) if ym else ""
        # auteur “probable” = premier mot Capitalisé du début
        am = re.search(r"^([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ\-']+)", os.path.splitext(base)[0])
        first_author = norm(am.group(1)) if am else ""
        out.append({"path": f, "base": base, "norm": n, "year": year, "first_author": first_author})
    return out

PDF_INDEX = list_pdf_files(PDF_TO_SCAN_DIR)

# --- Matching ----------------------------------------------------------------

def score_match(parsed_ref, pdf_meta):
    score = 0.0
    # année exacte
    if parsed_ref["year"] and parsed_ref["year"] == pdf_meta["year"]:
        score += 2.0
    # premier auteur
    if parsed_ref["authors"]:
        a1 = parsed_ref["authors"][0]
        if a1 and (a1 in pdf_meta["norm"] or a1 == pdf_meta.get("first_author", "")):
            score += 1.5
    # second auteur éventuel
    if len(parsed_ref["authors"]) >= 2:
        a2 = parsed_ref["authors"][1]
        if a2 and a2 in pdf_meta["norm"]:
            score += 1.0
    # si plusieurs auteurs dans la ref, bonus si au moins 2 sont trouvés
    hits = sum(1 for a in parsed_ref["authors"] if a and a in pdf_meta["norm"])
    if hits >= 2:
        score += 0.5
    # fallback: similarité globale (faible poids)
    sim = difflib.SequenceMatcher(None, norm(parsed_ref["raw"]), pdf_meta["norm"]).ratio()
    score += 0.5 * sim  # 0 -> 0.5
    return score

def find_best_pdf_for_reference(ref_str, pdf_index=PDF_INDEX, min_score=2.2):
    p = parse_reference(ref_str)
    if not p["authors"] and not p["year"]:
        return None, 0.0
    best = None; best_s = -1.0
    for meta in pdf_index:
        s = score_match(p, meta)
        if s > best_s:
            best_s, best = s, meta
    if best_s >= min_score:
        return best, best_s
    return None, best_s

# --- Ouverture cross-platform -------------------------------------------------

def open_file_native(path: str):
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

# --- Pipeline principal : extrait, matche, ouvre ------------------------------

def Reference(doc, output_file="reference.txt"):
    nombre_de_pages = len(doc.pages)
    compteur = 0
    with open(output_file, "w", encoding="utf-8") as ref:
        ref.write("Liste des références et correspondances PDF\n\n")
        for i in range(nombre_de_pages):
            page = doc.pages[i]
            text = page.extract_text() or ""
            ref.write(f"Page {i+1}\n\n")
            codes = [c.replace("\n", " ") for c in TrouveReference(text)]
            for j, raw_ref in enumerate(codes, 1):
                compteur += 1
                meta, s = find_best_pdf_for_reference(raw_ref)
                if meta:
                    line = f"Référence {compteur} : {raw_ref}  -->  {meta['base']}  [score={s:.2f}]\n"
                    if OPEN_MATCHED_PDFS:
                        try:
                            open_file_native(meta["path"])
                        except Exception as e:
                            line += f"(ouverture échouée: {e})"
                else:
                    line = f"Référence {compteur} : {raw_ref}  -->  (AUCUN PDF TROUVÉ, meilleur score={s:.2f})\n"
                    print(f"Il y'a un problème avec référence '{raw_ref}' à la page {i + 1}")
                ref.write(line)
            ref.write("\n")

# Lance le traitement
if __name__ == "__main__":
    Reference(doc)
    print("Terminé. Voir 'reference.txt'.")
