from dotenv import load_dotenv 
load_dotenv()
from openai import OpenAI
import time
import json
import os
import pdfplumber
from mistralai import Mistral
from sentence_transformers import SentenceTransformer, util
import torch
import re
import logging

# Désactiver les logs verbeux
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# --- CONFIGURATION ---
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-small-latest"

# Configuration pour Ollama
# OLLAMA_URL = "http://localhost:11434/v1"
# OLLAMA_MODEL = "mistral" 

# try:
#     client = OpenAI(
#         base_url=OLLAMA_URL, # <--- C'est ICI la magie. On pointe vers votre PC.
#         api_key="ollama"     # <--- Clé bidon. La librairie en exige une, mais Ollama s'en fiche.
#     )
# except Exception as e:
#     print(f"Erreur: {e}")

# --- CHARGEMENT MODÈLES ---
try:
    print("Chargement du modèle sémantique local...")
    LOCAL_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print(f"Erreur SentenceTransformer: {e}")
    LOCAL_MODEL = None

try:
    if MISTRAL_API_KEY:
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    else:
        mistral_client = None
        print("ATTENTION: Pas de clé API Mistral configurée.")
except Exception as e:
    print(f"Erreur Client Mistral: {e}")
    mistral_client = None


# --- FONCTIONS UTILITAIRES (VOTRE VERSION) ---

def get_pdf_full_text(pdf_path: str) -> str:
    """Votre fonction robuste utilisant pdfplumber"""
    if not os.path.exists(pdf_path):
        return ""
    try:
        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Erreur lecture PDF {pdf_path}: {e}")
        return ""

def find_most_relevant_chunks(context: str, reference_text: str, model, top_k=7, chunk_size=600, overlap=150) -> list[str]:
    if not context or not reference_text or not model:
        return []

    reference_text = reference_text.replace("-\n", "")
    words = reference_text.split()
    chunks = []
    
    word_chunk_size = int(chunk_size / 6) 
    word_overlap = int(overlap / 6)
    step = max(1, word_chunk_size - word_overlap)

    for i in range(0, len(words), step):
        chunk_words = words[i : i + word_chunk_size]
        chunk_text = " ".join(chunk_words)
        if len(chunk_text) > 50:
            chunks.append(chunk_text)
    
    if not chunks:
        return []

    try:
        context_embedding = model.encode(context, convert_to_tensor=True)
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        cosine_scores = util.cos_sim(context_embedding, chunk_embeddings)[0]
        
        k_val = min(top_k, len(chunks))
        top_k_results = torch.topk(cosine_scores, k=k_val)
        return [chunks[idx] for idx in top_k_results.indices]
    except Exception as e:
        print(f"Erreur sémantique locale : {e}")
        return []

def get_mistral_verification(context: str, relevant_chunks: list[str]) -> dict:
    if not relevant_chunks or not mistral_client:
        return {"score": 0.0, "justification": "Pas de chunks ou client Mistral non configuré."}

    chunks_text = "\n\n---\n\n".join(relevant_chunks)

    system_prompt = """
    Tu es un assistant de recherche expert chargé de valider des bibliographies scientifiques.
    Ta mission : Trouver le lien logique entre une "Citation" (ce que dit l'auteur) et les "Extraits Source".
    
    Utilise strictement cette échelle de notation :
    - 0.1 (ROUGE - HORS SUJET)
    - 0.5 (ORANGE - PLAUSIBLE MAIS VAGUE)
    - 0.8 (VERT - VALIDÉ)
    - 1.0 (PARFAIT)

    Répondez UNIQUEMENT au format JSON : {"score": 0.0, "justification": "..."}
    """
    
    user_prompt = f"Contexte: {context}\nExtraits: {chunks_text}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Boucle de retry (3 tentatives) - Gardée de votre version
    for i in range(1, 4): 
        try:
            chat_response = mistral_client.chat.complete(
                model=MISTRAL_MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = chat_response.choices[0].message.content
            result = json.loads(content)
            return {
                "score": float(result.get("score", 0.0)),
                "justification": str(result.get("justification", "Erreur format."))
            }
        except Exception as e:
            wait_time = i * 20
            print(f"Tentative {i} échouée: {e}. Attente {wait_time}s...")
            time.sleep(wait_time)
            
    return {"score": 0.0, "justification": "Erreur API Mistral persistante."}



# --- FONCTION D'INTERROGATION OLLAMA (Remplace get_mistral_verification) ---

# def query_ollama_mistral(context: str, relevant_chunks: list[str]) -> dict:
#     """
#     Envoie le contexte et les extraits à Ollama (Mistral local) pour validation.
#     """
#     # Sécurité : Si pas de chunks ou pas de client, on arrête
#     if not relevant_chunks or not client:
#         return {"score": 0.0, "justification": "Pas de chunks ou Ollama non connecté."}

#     # On colle les morceaux de texte avec des séparateurs visibles
#     chunks_text = "\n\n---\n\n".join(relevant_chunks)

#     # Le Prompt Système : On force le rôle et le format JSON
#     system_prompt = """
#     Tu es un expert en vérification bibliographique.
#     Analyse si le texte du PDF supporte la citation fournie.
#     Réponds UNIQUEMENT au format JSON strict.

#     Format attendu :
#     {
#         "score": 0.0 à 1.0,
#         "justification": "Une phrase courte d'explication en français."
#     }

#     Échelle de notation :
#     - 0.0 : Le PDF ne parle pas du tout du sujet ou contredit la citation.
#     - 0.5 : Le sujet est mentionné mais le lien est flou ou partiel.
#     - 0.8 : Le PDF supporte clairement la citation, même si ce n'est pas totalement parfait.
#     - 1.0 : Le PDF confirme explicitement et clairement la citation.
#     """
    
#     # Le Prompt Utilisateur : Les données brutes
#     user_prompt = f"""
#     CITATION À VÉRIFIER :
#     "{context}"

#     EXTRAITS TROUVÉS DANS LE PDF :
#     {chunks_text}

#     Consigne : Est-ce que ces extraits confirment la citation ?
#     """

#     try:
#         # Appel à l'API locale (Ollama)
#         response = client.chat.completions.create(
#             model=OLLAMA_MODEL,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0.1, # Créativité au minimum pour être factuel
#             response_format={"type": "json_object"} # Force le mode JSON valide
#         )
        
#         # Récupération de la réponse textuelle
#         content = response.choices[0].message.content
        
#         # Tentative de conversion du texte en Dictionnaire Python (JSON Parsing)
#         try:
#             result = json.loads(content)
#             return {
#                 "score": float(result.get("score", 0.0)),
#                 "justification": str(result.get("justification", "Pas de justification fournie."))
#             }
#         except json.JSONDecodeError:
#             # Si Mistral a mal formé son JSON (rare avec le mode json_object mais possible)
#             return {"score": 0.0, "justification": f"Erreur format JSON reçu: {content[:50]}..."}
            
#     except Exception as e:
#         # Si Ollama est éteint ou plante
#         return {"score": 0.0, "justification": f"Erreur connexion Ollama: {str(e)}"}


# --- NOUVELLE FONCTION REQUISE PAR L'INTERFACE ---

def verify_jobs_stream(jobs: list[dict]):
    """
    Fonction générateur utilisée par main.py.
    Elle remplace run_verification() pour l'interface graphique.
    """
    if not LOCAL_MODEL:
        yield 0, {"error": "Modèle local non chargé"}
        return
    
    ##Version Ollama : On vérifie que le client est opérationnel avant de lancer la boucle
    # # Petit check de sécurité
    # try:
    #     client.models.list()
    # except Exception:
    #     print("ATTENTION: Ollama ne semble pas tourner. Lancez 'ollama run mistral'.")
    

    for i, job in enumerate(jobs):
        # On ignore les jobs qui ont échoué à l'étape 1
        if job.get("status") == "PDF_NOT_FOUND":
            job["mistral_score"] = 0.0
            job["mistral_justification"] = "Fichier PDF non trouvé (Etape 1)."
            yield i, job
            continue

        # Lecture
        full_text = get_pdf_full_text(job["matched_pdf_path"])
        if not full_text:
            job["mistral_score"] = 0.0
            job["mistral_justification"] = "Impossible d'extraire le texte du PDF."
            yield i, job
            continue

        # Recherche
        relevant_chunks = find_most_relevant_chunks(
            job["citation_context"],
            full_text,
            LOCAL_MODEL
        )

        # Vérification
        if relevant_chunks:
            res = get_mistral_verification(job["citation_context"], relevant_chunks)
            #res = query_ollama_mistral(job["citation_context"], relevant_chunks) ####Version Ollama
        else:
            res = {"score": 0.0, "justification": "Aucun passage pertinent trouvé localement."}

        job["mistral_score"] = res["score"]
        job["mistral_justification"] = res["justification"]

        yield i, job
        time.sleep(0.1) # Petite pause UI

if __name__ == "__main__":
    pass