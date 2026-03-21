
import re
from typing import List, Dict, Optional

import torch
from sentence_transformers import util
from sentence_transformers.cross_encoder import CrossEncoder


def normalize_pdf_text(text: str) -> str:
    """
    Light cleanup for PDF-extracted text.
    """
    if not text:
        return ""

    # Fix common PDF hyphenation across line breaks
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Collapse extra spaces but preserve paragraph boundaries
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def split_text_into_semantic_chunks(
    text: str,
    target_words: int = 180,
    overlap_words: int = 40,
    min_words: int = 40,
) -> List[Dict]:
    """
    Chunk by paragraphs first, then merge into semantic windows.
    This is better than naive fixed-size character chunking for scientific PDFs.
    """
    text = normalize_pdf_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks: List[Dict] = []

    current_words: List[str] = []
    chunk_id = 0

    for paragraph in paragraphs:
        para_words = paragraph.split()

        # Skip very tiny garbage fragments
        if len(para_words) < 8:
            continue

        # If a paragraph is itself very large, split it into overlapping windows
        if len(para_words) > target_words * 2:
            if current_words:
                chunk_text = " ".join(current_words).strip()
                if len(chunk_text.split()) >= min_words:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "word_count": len(chunk_text.split()),
                    })
                    chunk_id += 1
                current_words = []

            step = max(1, target_words - overlap_words)
            for i in range(0, len(para_words), step):
                window = para_words[i:i + target_words]
                if len(window) >= min_words:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": " ".join(window).strip(),
                        "word_count": len(window),
                    })
                    chunk_id += 1
            continue

        # Try to accumulate paragraphs into a chunk close to target_words
        if len(current_words) + len(para_words) <= target_words:
            current_words.extend(para_words)
        else:
            if len(current_words) >= min_words:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": " ".join(current_words).strip(),
                    "word_count": len(current_words),
                })
                chunk_id += 1

                # overlap from previous chunk
                overlap = current_words[-overlap_words:] if overlap_words > 0 else []
                current_words = overlap + para_words
            else:
                current_words.extend(para_words)

    if len(current_words) >= min_words:
        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(current_words).strip(),
            "word_count": len(current_words),
        })

    return chunks


def load_reranker(model_name: str = "BAAI/bge-reranker-v2-m3") -> CrossEncoder:
    """
    Multilingual reranker that works well for French/English scientific text.
    """
    return CrossEncoder(model_name)


def retrieve_and_rerank_chunks(
    context: str,
    reference_text: str,
    embedding_model,
    reranker_model: Optional[CrossEncoder] = None,
    initial_top_k: int = 12,
    final_top_k: int = 5,
    target_words: int = 180,
    overlap_words: int = 40,
    min_words: int = 40,
) -> List[Dict]:
    """
    1) Build semantic chunks
    2) Retrieve best candidates with embeddings
    3) Rerank retrieved candidates with a cross-encoder

    Returns a list of dicts sorted by final relevance.
    """
    if not context or not reference_text:
        return []

    chunks = split_text_into_semantic_chunks(
        reference_text,
        target_words=target_words,
        overlap_words=overlap_words,
        min_words=min_words,
    )
    if not chunks:
        return []

    chunk_texts = [c["text"] for c in chunks]

    # Fast dense retrieval
    context_embedding = embedding_model.encode(
        context,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )

    cosine_scores = util.cos_sim(context_embedding, chunk_embeddings)[0]
    initial_top_k = min(initial_top_k, len(chunks))
    top_results = torch.topk(cosine_scores, k=initial_top_k)

    retrieved = []
    for score, idx in zip(top_results.values.tolist(), top_results.indices.tolist()):
        item = dict(chunks[idx])
        item["embedding_score"] = float(score)
        item["rerank_score"] = None
        item["final_score"] = float(score)
        retrieved.append(item)

    # Precise reranking
    if reranker_model is not None and retrieved:
        pairs = [(context, item["text"]) for item in retrieved]
        rerank_scores = reranker_model.predict(pairs)

        for item, rr_score in zip(retrieved, rerank_scores):
            rr_score = float(rr_score)
            item["rerank_score"] = rr_score

            # Use reranker as the primary score for final ordering
            # Keep embedding score as a tie-breaker / traceability feature
            item["final_score"] = rr_score

        retrieved.sort(
            key=lambda x: (x["final_score"], x["embedding_score"]),
            reverse=True,
        )
    else:
        retrieved.sort(key=lambda x: x["embedding_score"], reverse=True)

    return retrieved[:min(final_top_k, len(retrieved))]


def get_relevant_chunk_texts(
    context: str,
    reference_text: str,
    embedding_model,
    reranker_model: Optional[CrossEncoder] = None,
    initial_top_k: int = 12,
    final_top_k: int = 5,
    target_words: int = 180,
    overlap_words: int = 40,
    min_words: int = 40,
) -> List[str]:
    """
    Convenience wrapper if you only need the final chunk texts.
    """
    results = retrieve_and_rerank_chunks(
        context=context,
        reference_text=reference_text,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        initial_top_k=initial_top_k,
        final_top_k=final_top_k,
        target_words=target_words,
        overlap_words=overlap_words,
        min_words=min_words,
    )
    return [item["text"] for item in results]
