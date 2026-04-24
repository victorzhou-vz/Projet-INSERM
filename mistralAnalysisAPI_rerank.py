from dotenv import load_dotenv
load_dotenv()

import time
import json
import os
import pdfplumber
from mistralai.client import Mistral
from sentence_transformers import SentenceTransformer
import requests
import logging

from semantic_retrieval import retrieve_and_rerank_chunks, load_reranker

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


# 1) Configuration
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("ATTENTION : Configurez votre MISTRAL_API_KEY avant de lancer.")
    exit()

MISTRAL_MODEL = "mistral-small-latest"
OUTPUT_JSON_RESULTS = "data/verification_results.json"

# Retrieval parameters
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
INITIAL_TOP_K = 12
FINAL_TOP_K = 5
CHUNK_TARGET_WORDS = 180
CHUNK_OVERLAP_WORDS = 40
CHUNK_MIN_WORDS = 40


# 2) Models and client
try:
    print("Chargement du modèle d'embeddings...")
    LOCAL_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Modèle d'embeddings chargé.")

    print("Chargement du reranker...")
    RERANKER_MODEL = load_reranker(RERANKER_MODEL_NAME)
    print("Reranker chargé.")
except Exception as e:
    print(f"Erreur chargement modèles Sentence Transformers / Reranker: {e}")
    print("Veuillez installer 'sentence-transformers' et 'torch'.")
    exit()

try:
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
except Exception as e:
    print(f"Erreur client Mistral: {e}")
    exit()


# 3) Utilities
def get_pdf_full_text(pdf_path: str) -> str:
    """Extrait le texte complet avec pdfplumber."""
    if not os.path.exists(pdf_path):
        print(f"  Erreur: Fichier PDF non trouvé à {pdf_path}")
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
        print(f"  Erreur lecture PDF: {e}")
        return ""


def find_most_relevant_chunks(context: str, reference_text: str) -> list[dict]:
    """
    New pipeline:
    1) semantic chunking
    2) dense retrieval with embeddings
    3) reranking with a cross-encoder
    """
    try:
        return retrieve_and_rerank_chunks(
            context=context,
            reference_text=reference_text,
            embedding_model=LOCAL_MODEL,
            reranker_model=RERANKER_MODEL,
            initial_top_k=INITIAL_TOP_K,
            final_top_k=FINAL_TOP_K,
            target_words=CHUNK_TARGET_WORDS,
            overlap_words=CHUNK_OVERLAP_WORDS,
            min_words=CHUNK_MIN_WORDS,
        )
    except Exception as e:
        print(f"  Erreur pendant la recherche/reranking local : {e}")
        return []


OLLAMA_MODEL = "mistral:instruct"  # Nom du modèle Mistral installé localement via Ollama


def get_mistral_verification(context: str, relevant_chunks: list[str], should_abort=None, use_ollama: bool = False) -> dict:
    if not relevant_chunks:
        return {"score": 0.0, "justification": "Aucun morceau pertinent trouvé localement."}

    chunks_text = "\n\n---\n\n".join(relevant_chunks)

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

    # --- Chemin Ollama (modèle local) ---
    if use_ollama:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        for i in range(1, 4):
            try:
                result = ollama_generate_json(full_prompt, model=OLLAMA_MODEL, should_abort=should_abort)
                return {
                    "score": float(result.get("score", 0.0)),
                    "justification": str(result.get("justification", "Erreur format JSON."))
                }
            except VerificationAborted:
                raise
            except Exception as e:
                wait_time = i * 5
                print(f"  ⚠️ Ollama tentative {i}/3 échouée (Erreur: {e}). Pause de {wait_time}s...")
                time.sleep(wait_time)
        print("  Échec définitif Ollama après 3 tentatives.")
        return {"score": 0.0, "justification": "Erreur Ollama persistante."}

    # --- Chemin Mistral API (cloud) ---
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for i in range(1, 4):
        try:
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
            wait_time = i * 20
            print(f"  ⚠️ Tentative {i}/3 échouée (Erreur: {e}). Pause de {wait_time}s...")
            time.sleep(wait_time)

    print("  Échec définitif après 3 tentatives.")
    return {"score": 0.0, "justification": "Erreur API persistante (Trop de requêtes)."}


def verify_jobs_stream(jobs, should_abort=None, use_ollama: bool = False):
    """
    UI-friendly streaming verification:
    yields (index, updated_job_dict) for each job processed.
    """
    for i, job in enumerate(jobs):
        if should_abort is not None and should_abort():
            raise VerificationAborted()

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

        retrieved_chunks = find_most_relevant_chunks(
            job["citation_context"],
            full_text
        )

        relevant_chunks = [item["text"] for item in retrieved_chunks]
        job["retrieved_chunks"] = [
            {
                "chunk_id": item["chunk_id"],
                "embedding_score": item["embedding_score"],
                "rerank_score": item["rerank_score"],
                "final_score": item["final_score"],
                "text_preview": limit_text(item["text"], 500),
            }
            for item in retrieved_chunks
        ]

        if relevant_chunks:
            mistral_result = get_mistral_verification(
                job["citation_context"],
                relevant_chunks,
                should_abort=should_abort,
                use_ollama=use_ollama,
            )
        else:
            mistral_result = {
                "score": 0.0,
                "justification": "Aucun passage pertinent trouvé dans le document par le modèle local."
            }

        job["mistral_score"] = float(mistral_result["score"])
        job["mistral_justification"] = str(mistral_result["justification"])

        yield i, job
        time.sleep(3)

    with open(OUTPUT_JSON_RESULTS, "w") as f:
        json.dump(jobs, f, indent=4, ensure_ascii=False)
