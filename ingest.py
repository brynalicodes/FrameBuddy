import asyncio
import aiohttp
import json
from pathlib import Path
from aiohttp import ClientTimeout
from asynciolimiter import Limiter

import numpy as np
import torch
from PIL import Image
import io

from transformers import SiglipVisionModel, SiglipImageProcessor

# Load SigLIP image tower correctly
image_processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-384")
image_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-384")


final_ids = None
limiter = Limiter(80)  # 80 requests per second? edit:changing the rate limit does not help
REQUEST_TIMEOUT = 10
RETRIES = 3

DATA_DIR = Path("src/data")
METADATA_PATH = DATA_DIR / "metadata.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

async def fetch(url, session):
    await limiter.wait() 
    async with session.get(url) as resp:
        return await resp.text()

async def main(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(url, session) for url in urls]
        return await asyncio.gather(*tasks)


def embed_image_from_bytes(img_bytes: bytes) -> np.ndarray:
    """
    Take raw image bytes, return a (1, 768) float32 L2-normalized embedding.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = image_processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = image_model(pixel_values=inputs.pixel_values)
        emb = outputs.pooler_output  # (1, 768)

    emb = emb.cpu().numpy().astype("float32")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb  # shape (1, 768)

OBJECTS_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"
SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q=painting"

MAX_RESULTS = 500 # this will limit the num of kept results but not increase them
CONCURRENCY = 25

semaphore = asyncio.Semaphore(CONCURRENCY)


async def fetch_json(session, url):
    headers = {"User-Agent": "FrameBuddy/1.0"} # this should help w the rate limiting
    for _ in range(RETRIES):
        try:
            async with semaphore:
                async with session.get(url, headers=headers, timeout=REQUEST_TIMEOUT) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.json()
        except:
            await asyncio.sleep(0.5)
    return None


async def fetch_bytes(session: aiohttp.ClientSession, url: str):
    for _ in range(RETRIES):
        try:
            async with semaphore:
                async with session.get(url, timeout=REQUEST_TIMEOUT) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.read()
        except Exception:
            await asyncio.sleep(0.5)
    return None

async def get_ids_with_images(session):
    print("Calling search endpoint…")
    async with session.get(SEARCH_URL, timeout=60) as resp:
        if resp.status != 200:
            print("Search endpoint failed:", resp.status)
            return set()
        data = await resp.json()
        print("Search returned:", len(data.get("objectIDs", [])))
        return set(data.get("objectIDs", []))


async def get_ids_in_departments(session, departments):
    dept_ids = set()
    for d in departments:
        url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects?departmentIds={d}"
        data = await fetch_json(session, url)
        if data and data.get("objectIDs"):
            dept_ids.update(data["objectIDs"])
    return dept_ids


async def process_object(session, oid):
    data = await fetch_json(session, OBJECTS_URL.format(oid))
    if not data:
        return None

    #Reject non‑public‑domain objects immediately for legal reasons and all that
        #when i remove this i get above 100 for kept objects but it still doesn't increase as i iterate
    if data.get("isPublicDomain") is not True:
        return None


    # If both are empty, try additionalImages
    img_url = (
        data.get("primaryImageSmall")
        or data.get("primaryImage")
        or (data.get("additionalImages") or [None])[0]
    )

    img_bytes = await fetch_bytes(session, img_url)
    if not img_bytes:
        return None
    
    return data, img_bytes
    

    #print(oid, repr(data.get("primaryImage")), repr(data.get("primaryImageSmall"))) # see its getting way more urls than its admitting to




async def main():
    async with aiohttp.ClientSession() as session:

        # 1. All IDs with images
        has_images = await get_ids_with_images(session)
        print("IDs with images:", len(has_images))

        # 2. All IDs in chosen departments
        departments = [6, 18, 11, 14]  # example
        dept_ids = await get_ids_in_departments(session, departments)
        print("IDs in departments:", len(dept_ids))

        # 3. Intersection
        final_ids = list(has_images.intersection(dept_ids))
        #random.shuffle(final_ids) # shuffled because I thought maybe images were frontloaded.
        print("Final usable IDs:", len(final_ids))
        results = []
        tasks = []

        results_metadata = []
        results_embeddings = []

        tasks = []

        for idx, oid in enumerate(final_ids):
            tasks.append(process_object(session, oid))

            if len(tasks) >= CONCURRENCY:
                batch = await asyncio.gather(*tasks)
                tasks.clear()

                for item in batch:
                    if item:
                        meta, img_bytes = item
                        emb = embed_image_from_bytes(img_bytes)
                        results_embeddings.append(emb[0])
                        results_metadata.append(meta)

                        if len(results_metadata) >= MAX_RESULTS:
                            break

            if len(results_metadata) >= MAX_RESULTS:
                break

            if idx % 25 == 0:
                print(f"Processed {idx} objects... kept {len(results_metadata)}")


        #leftover tasks
        if tasks and len(results_metadata) < MAX_RESULTS:
            batch = await asyncio.gather(*tasks)
            for item in batch:
                if item and len(results_metadata) < MAX_RESULTS:
                    meta, img_bytes = item
                    emb = embed_image_from_bytes(img_bytes)
                    results_embeddings.append(emb[0])
                    results_metadata.append(meta)

        print("Final kept objects:", len(results_metadata))

        # Save metadata
        with open(METADATA_PATH, "w", encoding="utf-8") as out:
            for item in results_metadata:
                out.write(json.dumps(item) + "\n")

        # Save embeddings
        np.save(EMBEDDINGS_PATH, np.array(results_embeddings, dtype=np.float32))

        print(f"Saved metadata to {METADATA_PATH}")
        print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    
        print(final_ids[:20])


if __name__ == "__main__":
    asyncio.run(main())