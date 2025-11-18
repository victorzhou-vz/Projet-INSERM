from dotenv import load_dotenv  # Nouvelle ligne
load_dotenv()
import time
import json
import os
from PyPDF2 import PdfReader
from mistralai import Mistral
#from mistralai.models.messages import ChatMessage
from sentence_transformers import SentenceTransformer, util
import torch
import re

# --- 1. Configuration ---

# Mettez votre clé API Mistral ici
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("ATTENTION : Configurez votre MISTRAL_API_KEY avant de lancer.")
    exit() # Décommentez pour arrêter le script si la clé n'est pas définie

MISTRAL_MODEL = "mistral-small-latest" # Ou "mistral-large-latest"
INPUT_JSON = "verification_jobs.json"
OUTPUT_REPORT = "verification_report.txt"
OUTPUT_JSON_RESULTS = "verification_results.json"

# --- 2. Modèles et Clients (chargés une seule fois) ---

try:
    print("Chargement du modèle sémantique local (embeddings)...")
    LOCAL_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Modèle local chargé.")
except Exception as e:
    print(f"Erreur chargement SentenceTransformer: {e}")
    print("Veuillez installer 'sentence-transformers' et 'torch'.")
    exit()

try:
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
except Exception as e:
    print(f"Erreur client Mistral: {e}")
    exit()

# --- 3. Fonctions Utilitaires ---

def get_pdf_full_text(pdf_path: str) -> str:
    """Extrait le texte complet d'un fichier PDF."""
    if not os.path.exists(pdf_path):
        print(f"  Erreur: Fichier PDF non trouvé à {pdf_path}")
        return ""
    try:
        reader = PdfReader(pdf_path)
        full_text = []
        for page in reader.pages:
            full_text.append(page.extract_text() or "")
        return "\n".join(full_text)
    except Exception as e:
        print(f"  Erreur lors de la lecture de {pdf_path}: {e}")
        return ""

def find_most_relevant_chunks(context: str, reference_text: str, model, top_k=3, chunk_size=400, overlap=50) -> list[str]:
    """
    (Fonction de l'étape précédente)
    Trouve les 'top_k' morceaux les plus pertinents d'un long texte.
    J'ai réduit chunk_size pour être plus précis.
    """
    if not context or not reference_text:
        return []

    words = re.split(r'\s+', reference_text) # Sépare sur les espaces
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_text = " ".join(words[i : i + chunk_size])
        chunks.append(chunk_text)
    
    if not chunks:
        return []

    try:
        context_embedding = model.encode(context, convert_to_tensor=True)
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        cosine_scores = util.cos_sim(context_embedding, chunk_embeddings)[0]
        top_k_results = torch.topk(cosine_scores, k=min(top_k, len(chunks)))
        return [chunks[idx] for idx in top_k_results.indices]
        
    except Exception as e:
        print(f"  Erreur pendant la recherche sémantique locale : {e}")
        return []

def get_mistral_verification(context: str, relevant_chunks: list[str]) -> dict:
    """
    (Fonction de l'étape précédente)
    Demande à Mistral de noter la pertinence d'une citation.
    """
    if not relevant_chunks:
        return {"score": 0.0, "justification": "Aucun morceau pertinent trouvé dans le PDF de référence via le scan local."}

    chunks_text = "\n\n---\n\n".join(relevant_chunks)

    system_prompt = """
    Vous êtes un assistant de recherche expert vérifiant la fidélité des citations.
    Je vous fournis un "Contexte de Citation" (la phrase qui cite) et des "Extraits Pertinents" (les passages les plus similaires de l'article cité).
    Votre tâche est de juger si le Contexte est une affirmation fidèle basée sur les Extraits.
    
    Fournissez un "score" (0.0 à 1.0) et une "justification" (1-2 phrases).
    - 1.0 : Le Contexte est une affirmation directe et vérifiable des Extraits.
    - 0.8 : Le Contexte est une paraphrase ou un résumé fidèle.
    - 0.5 : Le Contexte est lié, mais l'affirmation principale n'est pas explicitement soutenue.
    - 0.1 : Les Extraits sont sur le même sujet mais ne soutiennent pas du tout l'affirmation du Contexte.
    
    Répondez *uniquement* en JSON : {"score": 0.0, "justification": "..."}
    """

    user_prompt = f"""
    **Contexte de Citation:**
    {context}

    **Extraits Pertinents de l'article cité:**
    {chunks_text}
    """
    for i in range (1, 4) : 
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            chat_response = mistral_client.chat.complete(
                model=MISTRAL_MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )
            response_content = chat_response.choices[0].message.content
            result = json.loads(response_content)
            return {
                "score": float(result.get("score", 0.0)),
                "justification": str(result.get("justification", "Erreur format JSON."))
            }
        except Exception as e:
            print(f"  Attempt n°{i} - Erreur API Mistral : {e}")
            print(f"Attente de {10*i} secondes avant nouvelle tentative...")
            time.sleep(10*i)
        

    print("Échec après 3 tentatives.")
    return {"score": 0.0, "justification": f"Erreur API: {e}"}
        
        

# --- 4. Processus Principal ---

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
        time.sleep(3) # Pause pour éviter de surcharger l'API

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


if __name__ == "__main__":
    run_verification()