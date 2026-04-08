from dotenv import load_dotenv 
load_dotenv()
import time
import json
import os
from PyPDF2 import PdfReader
import pdfplumber
from mistralai import Mistral
import mistralAnalysisAPI as mistralAnalysis
from sentence_transformers import SentenceTransformer, util
import torch
import re
import requests
import sys

import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)


class VerificationAborted(Exception):
    """Raised when user requests to stop verification."""
    pass


def ollama_generate_json(prompt: str, model: str, should_abort=None, timeout: int = 300) -> dict:
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Ollama connection failed: {e}") from e

    # If server returns an error, print the body (very important!)
    if resp.status_code >= 400:
        try:
            body = resp.text
        except Exception:
            body = "<could not read response body>"
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {body}")

    chunks = []
    try:
        for line in resp.iter_lines(decode_unicode=True):
            if should_abort is not None and should_abort():
                resp.close()
                raise VerificationAborted()

            if not line:
                continue

            obj = json.loads(line)

            if "response" in obj:
                chunks.append(obj["response"])

            if obj.get("done") is True:
                break
    finally:
        try:
            resp.close()
        except Exception:
            pass

    text = "".join(chunks).strip()
    return json.loads(text)

def limit_text(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    return s[:max_chars]

#1 Configuration

if getattr(sys, 'frozen', False):
    # Si le code est compilé par PyInstaller
    dossier_actuel = os.path.dirname(sys.executable)
else:
    # Si le code est lancé normalement via Python
    dossier_actuel = os.path.dirname(os.path.abspath(__file__))

chemin_env = os.path.join(dossier_actuel, '.env')
load_dotenv(chemin_env)

# Mettez votre clé API Mistral ici
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("ATTENTION : Configurez votre MISTRAL_API_KEY avant de lancer.")
    sys.exit(1) # Décommentez pour arrêter le script si la clé n'est pas définie


MISTRAL_MODEL = "mistral-small-latest" # Ou "mistral-large-latest"
# MISTRAL_MODEL = "mistral:instruct" 
INPUT_JSON = "verification_jobs.json"
OUTPUT_REPORT = "verification_report.txt"
OUTPUT_JSON_RESULTS = "verification_results.json"

#2 Modèles et Clients

try:
    print("Chargement du modèle sémantique local (embeddings)...")
    LOCAL_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Modèle local chargé.")
except Exception as e:
    print(f"Erreur chargement SentenceTransformer: {e}")
    print("Veuillez installer 'sentence-transformers' et 'torch'.")
    sys.exit(1)

try:
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
except Exception as e:
    print(f"Erreur client Mistral: {e}")
    sys.exit(1)

#3 Fonctions Utilitaires

def get_pdf_full_text(pdf_path: str) -> str:
    """Extrait le texte complet avec pdfplumber (gère mieux les colonnes)."""
    if not os.path.exists(pdf_path):
        print(f"  Erreur: Fichier PDF non trouvé à {pdf_path}")
        return ""
    
    try:
        full_text = []
        # On ouvre avec pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
        return "\n".join(full_text)
        
    except Exception as e:
        print(f"  Erreur lecture PDF: {e}")
        return ""
    


def find_most_relevant_chunks(context: str, reference_text: str, model, top_k=7, chunk_size=600, overlap=150) -> list[str]:
    """
    Découpage par mots avec chevauchement (Overlap).
    top_k augmenté à 7 pour donner plus de contexte à Mistral.
    """
    if not context or not reference_text:
        return []

    # Petit nettoyage des césures (ex: "environne- ment")
    reference_text = reference_text.replace("-\n", "")
    
    words = reference_text.split() # On découpe par mots
    chunks = []
    
    # On convertit grossièrement la taille demandée (caractères) en nombre de mots
    # 1 mot ≈ 6 caractères (moyenne)
    word_chunk_size = int(chunk_size / 6) 
    word_overlap = int(overlap / 6)
    
    # On calcule le "pas" (step) pour avancer dans le texte
    step = max(1, word_chunk_size - word_overlap)

    for i in range(0, len(words), step):
        chunk_words = words[i : i + word_chunk_size]
        chunk_text = " ".join(chunk_words)
        # On garde seulement si le morceau est assez long (évite les numéros de page isolés)
        if len(chunk_text) > 50:
            chunks.append(chunk_text)
    
    if not chunks:
        return []

    try:
        context_embedding = model.encode(context, convert_to_tensor=True)
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        cosine_scores = util.cos_sim(context_embedding, chunk_embeddings)[0]
        
        # On prend les 'top_k' meilleurs (ex: 7)
        k_val = min(top_k, len(chunks))
        top_k_results = torch.topk(cosine_scores, k=k_val)
        return [chunks[idx] for idx in top_k_results.indices]
        
    except Exception as e:
        print(f"  Erreur pendant la recherche sémantique locale : {e}")
        return []
        

def get_mistral_verification(context: str, relevant_chunks: list[str], should_abort=None) -> dict:
    if not relevant_chunks:
        return {"score": 0.0, "justification": "Aucun morceau pertinent trouvé localement."}

    chunks_text = "\n\n---\n\n".join(relevant_chunks)

    # Note : J'ai simplifié le prompt pour économiser des tokens, mais le sens est le même
    system_prompt = """
    Tu es un assistant de recherche expert chargé de valider des bibliographies scientifiques.
    Ta mission : Trouver le lien logique entre une "Citation" (ce que dit l'auteur) et les "Extraits Source".
    
    IMPORTANT : Les auteurs scientifiques reformulent souvent, résument ou utilisent des synonymes.
    Ne sois pas trop littéral. Cherche le sens, pas juste les mots-clés exacts.
    
    Utilise strictement cette échelle de notation (Feux Tricolores) :
    
    - 0.1 (ROUGE - HORS SUJET) :
      L'affirmation est contredite ou le sujet des extraits n'a strictement RIEN à voir avec la citation.
      (Exemple : La citation parle d'alcool, l'extrait parle de climat).
      
    - 0.5 (ORANGE - PLAUSIBLE MAIS VAGUE) :
      Le sujet est le bon (ex: les deux parlent de coût social), mais l'affirmation précise (chiffre ou conclusion spécifique) n'est pas explicitement visible dans ces extraits.
      Cela peut être dû à un découpage du texte imparfait. On accorde le bénéfice du doute.
      
    - 0.8 (VERT - VALIDÉ) :
      L'affirmation est correcte. Elle correspond à une paraphrase, un résumé des conclusions, ou une donnée présente dans le texte (même formulée différemment).
      
    - 1.0 (PARFAIT) :
      Correspondance exacte ou quasi mot-pour-mot.
    
    Rappel : Si la citation résume la "tendance générale" des extraits, c'est un 0.8, pas un 0.5.

    Répondez UNIQUEMENT au format JSON : {"score": 0.0, "justification": "..."}

    Répondez en français (SINON VOUS MOURREZ), même si les extraits sont en anglais. L'important est de comprendre le sens global.
    """
    
    user_prompt = f"Contexte: {context}\nExtraits: {chunks_text}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # --- CORRECTION DE LA BOUCLE ---
    for i in range(1, 4): 
        try:
            chat_response = mistral_client.chat.complete(
                model=MISTRAL_MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )
            response_content = chat_response.choices[0].message.content
            result = json.loads(response_content)
            
            # SI SUCCÈS : On retourne le résultat (on sort de la fonction)
            return {
                "score": float(result.get("score", 0.0)),
                "justification": str(result.get("justification", "Erreur format JSON."))
            }
            
        except Exception as e:
            # SI ERREUR : On n'utilise PAS 'return' ici ! On attend et on continue la boucle.
            wait_time = i * 20 # 20s, 40s, 60s
            print(f"  ⚠️ Tentative {i}/3 échouée (Erreur: {e}). Pause de {wait_time}s...")
            time.sleep(wait_time)
            
    # SI ON ARRIVE ICI, c'est que la boucle est finie sans succès
    print("  Échec définitif après 3 tentatives.")
    return {"score": 0.0, "justification": "Erreur API persistante (Trop de requêtes)."}
      

#4 Processus Principal

def run_verification():
    # 1. Lire les tâches
    try:
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Fichier '{INPUT_JSON}' non trouvé.")
        print(f"Veuillez d'abord lancer 'match_references.py' (ou le nom de votre Fichier 1).")
        return
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier '{INPUT_JSON}' est corrompu ou vide.")
        return

    all_results = []
    
    print(f"Début de la vérification de {len(jobs)} tâches...")

    # 2. Traiter chaque tâche
    for i, job in enumerate(jobs):
        print(f"\n--- Traitement Tâche {i+1}/{len(jobs)}: {job['raw_citation']} (Page {job['page']}) ---")
        
        if job["status"] == "PDF_NOT_FOUND":
            print("  Statut : PDF_NOT_FOUND. Ignoré.")
            job["mistral_score"] = 0.0
            job["mistral_justification"] = "Le fichier PDF de référence n'a pas été trouvé lors de l'étape 1."
            all_results.append(job)
            continue
            
        # 2a. Lire le texte du PDF
        print(f"  Lecture de {job['pdf_filename']}...")
        full_text = get_pdf_full_text(job['matched_pdf_path'])
        if not full_text:
            job["mistral_score"] = 0.0
            job["mistral_justification"] = "Erreur lors de la lecture ou de l'extraction du texte du PDF."
            all_results.append(job)
            continue
            
        # 2b. Pré-filtrage sémantique local
        print("  Recherche locale des passages pertinents...")
        relevant_chunks = find_most_relevant_chunks(
            job['citation_context'],
            full_text,
            LOCAL_MODEL
        )
        
        # 2c. Appel API Mistral pour jugement
        if relevant_chunks:
            print(f"  Envoi de {len(relevant_chunks)} morceau(x) à Mistral pour vérification...")
            mistral_result = get_mistral_verification(
                job['citation_context'],
                relevant_chunks
            )
        else:
            print("  Aucun morceau sémantiquement pertinent trouvé localement.")
            mistral_result = {"score": 0.0, "justification": "Aucun passage pertinent trouvé dans le document par le modèle local."}

        job["mistral_score"] = mistral_result["score"]
        job["mistral_justification"] = mistral_result["justification"]
        all_results.append(job)
        
        print(f"  Score Sémantique : {job['mistral_score']}")
        print(f"  Justification : {job['mistral_justification']}")
        # time.sleep(3) # Pause pour éviter de surcharger l'API

    # 3. Sauvegarder les résultats
    
    # 3a. Sauvegarde en JSON (pour une utilisation future)
    with open(OUTPUT_JSON_RESULTS, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    # 3b. Sauvegarde en rapport TXT lisible
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("Rapport de Vérification des Citations\n")
        f.write("="*40 + "\n\n")
        
        for res in all_results:
            f.write(f"Référence (Page {res['page']}): {res['raw_citation']}\n")
            f.write(f"  Contexte : \"...{res['citation_context']}...\"\n")
            f.write(f"  Fichier PDF : {res.get('pdf_filename', 'N/A')}\n")
            f.write(f"  Score Sémantique (Mistral) : {res['mistral_score']:.2f}\n")
            f.write(f"  Justification : {res['mistral_justification']}\n")
            f.write("-" * 20 + "\n")
            
    print(f"\nTerminé ! Rapport complet sauvegardé dans '{OUTPUT_REPORT}'.")
    print(f"Résultats JSON bruts sauvegardés dans '{OUTPUT_JSON_RESULTS}'.")



def verify_jobs_stream(jobs, should_abort=None):    
    """
    UI-friendly streaming verification:
    yields (index, updated_job_dict) for each job processed.
    """
    for i, job in enumerate(jobs):
        if should_abort is not None and should_abort():
            raise VerificationAborted()
        # keep behavior consistent with your current run_verification()
        if job.get("status") == "PDF_NOT_FOUND":
            job["mistral_score"] = 0.0
            job["mistral_justification"] = "Le fichier PDF de référence n'a pas été trouvé lors de l'étape 1."
            yield i, job
            continue

        full_text = get_pdf_full_text(job["matched_pdf_path"])
        if not full_text:
            job["mistral_score"] = 0.0
            job["mistral_justification"] = "Erreur lors de la lecture ou de l'extraction du texte du PDF."
            yield i, job
            continue

        relevant_chunks = find_most_relevant_chunks(
            job["citation_context"],
            full_text,
            LOCAL_MODEL
        )

        if relevant_chunks:
            mistral_result = get_mistral_verification(
                job["citation_context"],
                relevant_chunks,
                should_abort=should_abort,
            )
        else:
            mistral_result = {
                "score": 0.0,
                "justification": "Aucun passage pertinent trouvé dans le document par le modèle local."
            }

        job["mistral_score"] = float(mistral_result["score"])
        job["mistral_justification"] = str(mistral_result["justification"])

        yield i, job
        time.sleep(0.2)  # small pause; UI runs in thread anyway


if __name__ == "__main__":
    run_verification()