# src/controller.py

from .preprocessing import preprocess_image
from .embedder import embed_image
from .indexer import load_faiss, search_faiss
from .retrieval import load_metadata, get_metadata_for_ids
from .prompts import build_reasoning_prompt
from .llm import run_qwen_vision
from .safety import apply_safety_filter


def run_framebuddy(image_path: str, top_k: int = 5):
    """
    Full FrameBuddy pipeline:
    image → preprocess → embed → FAISS → metadata → Qwen → safety → output
    """

    # 1. Preprocess
    img_tensor = preprocess_image(image_path)

    # 2. Embedding
    embedding = embed_image(img_tensor)

    # 3. FAISS retrieval
    index = load_faiss()
    ids, distances = search_faiss(index, embedding, top_k=top_k)

    # 4. Metadata lookup
    metadata_store = load_metadata()
    retrieved_metadata = get_metadata_for_ids(metadata_store, ids)

    # 5. Prompt construction
    prompt = build_reasoning_prompt(image_path, retrieved_metadata)

    # 6. Qwen reasoning
    raw_output = run_qwen_vision(prompt, image_path)

    # 7. Safety filter
    safe_output = apply_safety_filter(raw_output)

    # 8. Return final result
    return {
        "matches": retrieved_metadata,
        "model_output": safe_output,
        "raw_output": raw_output,
        "distances": distances,
    }
